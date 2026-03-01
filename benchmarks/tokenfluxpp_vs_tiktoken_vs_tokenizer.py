from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import shutil
import statistics
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = ROOT / "artifacts" / "benchmark_tokenizer.json"
DEFAULT_HF_TOKENIZER_PATH = ROOT / "artifacts" / "benchmark_tokenizer_hf.json"
SNIPPETS = [
    "TokenFlux++ focuses on low-latency tokenizer inference for production data pipelines.",
    "The runtime is implemented in C++ and exposed to Python through pybind bindings.",
    "Stable throughput under mixed document sizes is critical for pretraining workloads.",
    "Work stealing and bounded queues keep worker utilization high on multi-core CPUs.",
    "Pre-tokenization quality depends on tokenizer.json compatibility and text normalization.",
    "Batch encoding should remain deterministic across benchmark rounds for fair comparison.",
]


@dataclass
class BenchmarkResult:
    name: str
    latencies: list[float]
    token_counts: list[int]
    docs: int
    chars: int

    @property
    def mean_latency(self) -> float:
        return statistics.fmean(self.latencies)

    @property
    def std_latency(self) -> float:
        if len(self.latencies) <= 1:
            return 0.0
        return statistics.stdev(self.latencies)

    @property
    def mean_tokens(self) -> float:
        return statistics.fmean(self.token_counts)

    @property
    def docs_per_sec(self) -> float:
        return self.docs / self.mean_latency if self.mean_latency > 0 else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.chars / self.mean_latency if self.mean_latency > 0 else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.mean_tokens / self.mean_latency if self.mean_latency > 0 else 0.0

    @property
    def avg_tokens_per_doc(self) -> float:
        return self.mean_tokens / self.docs if self.docs > 0 else 0.0


@dataclass
class TimingResult:
    name: str
    latencies: list[float]

    @property
    def mean_latency(self) -> float:
        return statistics.fmean(self.latencies)

    @property
    def std_latency(self) -> float:
        if len(self.latencies) <= 1:
            return 0.0
        return statistics.stdev(self.latencies)


def _add_build_path_for_tokenflux_cpp() -> None:
    for suffix in ("pyd", "so"):
        for candidate in (ROOT / "build").rglob(f"tokenflux_cpp.{suffix}"):
            build_dir = str(candidate.parent)
            if build_dir not in sys.path:
                sys.path.insert(0, build_dir)
            return


def _load_tokenflux():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _add_build_path_for_tokenflux_cpp()
    try:
        return importlib.import_module("tokenflux")
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "Failed to import tokenflux. Build/install tokenflux_cpp first "
            "(example: `python -m pip install -e . --no-build-isolation`)."
        ) from exc


def _load_tiktoken():
    if importlib.util.find_spec("tiktoken") is None:
        raise RuntimeError("Missing dependency: tiktoken. Install with `python -m pip install tiktoken`.")
    return importlib.import_module("tiktoken")


def _load_hf_tokenizers():
    if importlib.util.find_spec("tokenizers") is None:
        raise RuntimeError("Missing dependency: tokenizers. Install with `python -m pip install tokenizers`.")
    return importlib.import_module("tokenizers")


def _generate_docs(num_docs: int, min_phrases: int, max_phrases: int, seed: int) -> list[str]:
    if min_phrases <= 0 or max_phrases < min_phrases:
        raise ValueError("Require: 0 < min_phrases <= max_phrases")
    rng = random.Random(seed)
    docs: list[str] = []
    for idx in range(num_docs):
        phrase_count = rng.randint(min_phrases, max_phrases)
        phrases = [SNIPPETS[rng.randrange(len(SNIPPETS))] for _ in range(phrase_count)]
        phrases.append(f"doc_id={idx}")
        docs.append(" ".join(phrases))
    return docs


def _parse_thread_points(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            lo_s, hi_s = text.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo <= 0 or hi <= 0:
                raise ValueError(f"Invalid thread range: {text}")
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
            continue
        value = int(text)
        if value <= 0:
            raise ValueError(f"Invalid thread value: {text}")
        values.append(value)
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    if not deduped:
        raise ValueError("No valid thread points.")
    return deduped


def _resolve_threads(value: str, arg_name: str, allow_auto: bool) -> tuple[str, int]:
    text = value.strip().lower()
    if allow_auto and text == "auto":
        return "auto", max(1, os.cpu_count() or 1)
    parsed = int(text)
    if parsed <= 0:
        raise ValueError(f"{arg_name} must be > 0")
    return str(parsed), parsed


def _resolve_trainer_kind(tf, trainer: str):
    key = trainer.strip().lower()
    if key not in {"byte_bpe", "bpe", "wordpiece", "unigram"}:
        raise ValueError("trainer must be one of: byte_bpe, bpe, wordpiece, unigram")
    return getattr(tf.TrainerKind, key)


def _bootstrap_tokenizer(
    tf,
    tokenizer_path: Path,
    docs: list[str],
    trainer: str,
    vocab_size: int,
    min_freq: int,
    min_pair_freq: int,
    threads: int,
) -> None:
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = artifacts_dir / f"tokenfluxpp_bench_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=False)
    try:
        sample = tmpdir / "bootstrap.jsonl"
        with sample.open("w", encoding="utf-8") as f:
            for text in docs:
                f.write(json.dumps({"text": text}))
                f.write("\n")

        cfg = tf.TrainConfig()
        cfg.trainer = _resolve_trainer_kind(tf, trainer)
        cfg.vocab_size = vocab_size
        cfg.min_freq = min_freq
        cfg.min_pair_freq = min_pair_freq
        cfg.threads = max(1, threads)
        cfg.resume = False
        cfg.chunk_dir = str(tmpdir / "chunks")
        cfg.output_json = str(tokenizer_path)
        cfg.output_vocab = str(tmpdir / "vocab.json")
        cfg.output_merges = str(tmpdir / "merges.txt")
        tf.train(cfg, [str(sample)])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def _temporary_env(name: str, value: Optional[str]) -> Iterator[None]:
    old = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _create_hf_train_setup(hf_tokenizers_mod, trainer_name: str, vocab_size: int, min_freq: int):
    models = importlib.import_module("tokenizers.models")
    trainers = importlib.import_module("tokenizers.trainers")
    pre_tokenizers = importlib.import_module("tokenizers.pre_tokenizers")
    decoders = importlib.import_module("tokenizers.decoders")
    token = "<|endoftext|>"
    key = trainer_name.strip().lower()

    if key == "byte_bpe":
        tok = hf_tokenizers_mod.Tokenizer(models.BPE(unk_token=token))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[token])
        return tok, tr
    if key == "bpe":
        tok = hf_tokenizers_mod.Tokenizer(models.BPE(unk_token=token))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[token])
        return tok, tr
    if key == "wordpiece":
        tok = hf_tokenizers_mod.Tokenizer(models.WordPiece(unk_token=token))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.decoder = decoders.WordPiece(prefix="##")
        tr = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            continuing_subword_prefix="##",
            special_tokens=[token],
        )
        return tok, tr
    if key == "unigram":
        tok = hf_tokenizers_mod.Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tr = trainers.UnigramTrainer(vocab_size=vocab_size, unk_token=token, special_tokens=[token])
        return tok, tr
    raise ValueError("Unsupported trainer for tokenizers compare.")


def _train_hf_tokenizer(
    hf_tokenizers_mod,
    tokenizer_path: Path,
    docs: list[str],
    trainer_name: str,
    vocab_size: int,
    min_freq: int,
    threads: int,
) -> None:
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tok, tr = _create_hf_train_setup(hf_tokenizers_mod, trainer_name, vocab_size, min_freq)
    with _temporary_env("RAYON_NUM_THREADS", str(max(1, threads))):
        tok.train_from_iterator(docs, trainer=tr)
    tok.save(str(tokenizer_path))


def _split_docs(docs: list[str], workers: int) -> list[list[str]]:
    workers = max(1, min(workers, len(docs)))
    base = len(docs) // workers
    rem = len(docs) % workers
    out: list[list[str]] = []
    start = 0
    for idx in range(workers):
        end = start + base + (1 if idx < rem else 0)
        chunk = docs[start:end]
        if chunk:
            out.append(chunk)
        start = end
    return out


def _make_tokenflux_runner(tf, tokenizer_path: Path, docs: list[str], workers: int) -> Callable[[], int]:
    workers = max(1, workers)
    if workers == 1 or len(docs) <= 1:
        tok = tf.Tokenizer(str(tokenizer_path))

        def run() -> int:
            batch = tok.encode_batch(docs, "", "", True)
            return sum(len(ids) for ids in batch)

        return run

    chunks = _split_docs(docs, workers)
    tokenizers = [tf.Tokenizer(str(tokenizer_path)) for _ in chunks]

    def run() -> int:
        total = 0
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(tok.encode_batch, chunk, "", "", True) for tok, chunk in zip(tokenizers, chunks)]
            for future in futures:
                total += sum(len(ids) for ids in future.result())
        return total

    return run


def _encode_tiktoken_batch(encoding, docs: list[str], threads: int) -> list[list[int]]:
    kwargs = {}
    if threads > 0:
        kwargs["num_threads"] = threads
    if hasattr(encoding, "encode_ordinary_batch"):
        try:
            return encoding.encode_ordinary_batch(docs, **kwargs)
        except TypeError:
            return encoding.encode_ordinary_batch(docs)
    try:
        return encoding.encode_batch(docs, disallowed_special=(), **kwargs)
    except TypeError:
        return encoding.encode_batch(docs, disallowed_special=())


def _encode_tiktoken_text(encoding, text: str) -> list[int]:
    if hasattr(encoding, "encode_ordinary"):
        return encoding.encode_ordinary(text)
    return encoding.encode(text, disallowed_special=())


def _decode_tiktoken_batch(encoding, batch_ids: list[list[int]], threads: int) -> list[str]:
    kwargs = {}
    if threads > 0:
        kwargs["num_threads"] = threads
    if hasattr(encoding, "decode_batch"):
        try:
            return encoding.decode_batch(batch_ids, **kwargs)
        except TypeError:
            return encoding.decode_batch(batch_ids)
    return [encoding.decode(ids) for ids in batch_ids]


def _decode_hf_batch(hf_tok, batch_ids: list[list[int]]) -> list[str]:
    if hasattr(hf_tok, "decode_batch"):
        try:
            return hf_tok.decode_batch(batch_ids)
        except TypeError:
            pass
    return [hf_tok.decode(ids) for ids in batch_ids]


def _make_tiktoken_runner(encoding, docs: list[str], threads: int) -> Callable[[], int]:
    def run() -> int:
        batch = _encode_tiktoken_batch(encoding, docs, threads)
        return sum(len(ids) for ids in batch)

    return run


def _uses_bytelevel_pretokenizer(tokenizer_path: Path) -> bool:
    def contains_bytelevel(node) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "ByteLevel":
                return True
            for value in node.values():
                if contains_bytelevel(value):
                    return True
            return False
        if isinstance(node, list):
            for item in node:
                if contains_bytelevel(item):
                    return True
        return False

    try:
        obj = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return contains_bytelevel(obj.get("pre_tokenizer"))


def _create_hf_tokenizer(hf_tokenizers_mod, tokenizer_path: Path):
    tok = hf_tokenizers_mod.Tokenizer.from_file(str(tokenizer_path))
    if _uses_bytelevel_pretokenizer(tokenizer_path):
        try:
            decoders = importlib.import_module("tokenizers.decoders")
            tok.decoder = decoders.ByteLevel()
        except Exception:
            pass
    return tok


def _make_hf_decode_runner(hf_tokenizers_mod, tokenizer_path: Path, batch_ids: list[list[int]], workers: int) -> Callable[[], int]:
    total_tokens = sum(len(ids) for ids in batch_ids)
    workers = max(1, workers)
    if workers == 1 or len(batch_ids) <= 1:
        tok = _create_hf_tokenizer(hf_tokenizers_mod, tokenizer_path)

        def run() -> int:
            _decode_hf_batch(tok, batch_ids)
            return total_tokens

        return run

    chunks = _split_docs(batch_ids, workers)
    toks = [_create_hf_tokenizer(hf_tokenizers_mod, tokenizer_path) for _ in chunks]

    def run() -> int:
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(_decode_hf_batch, tok, chunk) for tok, chunk in zip(toks, chunks)]
            for future in futures:
                future.result()
        return total_tokens

    return run


def _make_hf_encode_runner(hf_tokenizers_mod, tokenizer_path: Path, docs: list[str], workers: int) -> Callable[[], int]:
    workers = max(1, workers)
    if workers == 1 or len(docs) <= 1:
        tok = _create_hf_tokenizer(hf_tokenizers_mod, tokenizer_path)

        def run() -> int:
            encodings = tok.encode_batch(docs)
            return sum(len(item.ids) for item in encodings)

        return run

    chunks = _split_docs(docs, workers)
    toks = [_create_hf_tokenizer(hf_tokenizers_mod, tokenizer_path) for _ in chunks]

    def run() -> int:
        total = 0
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(tok.encode_batch, chunk) for tok, chunk in zip(toks, chunks)]
            for future in futures:
                encodings = future.result()
                total += sum(len(item.ids) for item in encodings)
        return total

    return run


def _run_benchmark(
    name: str,
    runner: Callable[[], int],
    warmup: int,
    repeat: int,
    docs: int,
    chars: int,
) -> BenchmarkResult:
    for _ in range(max(warmup, 0)):
        runner()
    latencies: list[float] = []
    token_counts: list[int] = []
    for _ in range(max(repeat, 1)):
        st = perf_counter()
        token_count = runner()
        latencies.append(perf_counter() - st)
        token_counts.append(token_count)
    return BenchmarkResult(name=name, latencies=latencies, token_counts=token_counts, docs=docs, chars=chars)


def _run_timing(name: str, runner: Callable[[], None], warmup: int, repeat: int) -> TimingResult:
    for _ in range(max(warmup, 0)):
        runner()
    latencies: list[float] = []
    for _ in range(max(repeat, 1)):
        st = perf_counter()
        runner()
        latencies.append(perf_counter() - st)
    return TimingResult(name=name, latencies=latencies)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _fmtf(value: float) -> str:
    return f"{value:,.4f}"


def _fmti(value: float) -> str:
    return f"{value:,.0f}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TokenFlux++ vs OpenAI tiktoken.")
    p.add_argument("--docs", type=int, default=200000)
    p.add_argument("--min-phrases", type=int, default=3)
    p.add_argument("--max-phrases", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260227)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--thread-points", default="1,2,4,8,16")
    p.add_argument("--sweep-warmup", type=int, default=1)
    p.add_argument("--sweep-repeat", type=int, default=2)
    p.add_argument("--tokenflux-tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--tokenflux-threads", default="auto", help="int or auto")
    p.add_argument("--tokenizers-tokenizer", type=Path, default=DEFAULT_HF_TOKENIZER_PATH)
    p.add_argument("--tokenizers-threads", default="auto", help="int or auto")
    p.add_argument("--trainer", default="byte_bpe")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--min-pair-freq", type=int, default=2)
    p.add_argument("--train-docs", type=int, default=5000)
    p.add_argument("--train-warmup", type=int, default=0)
    p.add_argument("--train-repeat", type=int, default=1)
    p.add_argument("--train-threads", default="auto", help="int or auto")
    p.add_argument("--bootstrap-docs", type=int, default=5000)
    p.add_argument("--bootstrap-threads", type=int, default=4)
    p.add_argument("--bootstrap-tokenizer", dest="bootstrap_tokenizer", action="store_true")
    p.add_argument("--no-bootstrap-tokenizer", dest="bootstrap_tokenizer", action="store_false")
    p.set_defaults(bootstrap_tokenizer=True)
    p.add_argument("--benchmark-train", dest="benchmark_train", action="store_true")
    p.add_argument("--no-benchmark-train", dest="benchmark_train", action="store_false")
    p.set_defaults(benchmark_train=True)
    p.add_argument("--benchmark-tokenizers-train", dest="benchmark_tokenizers_train", action="store_true")
    p.add_argument("--no-benchmark-tokenizers-train", dest="benchmark_tokenizers_train", action="store_false")
    p.set_defaults(benchmark_tokenizers_train=True)
    p.add_argument("--benchmark-decode", dest="benchmark_decode", action="store_true")
    p.add_argument("--no-benchmark-decode", dest="benchmark_decode", action="store_false")
    p.set_defaults(benchmark_decode=True)
    p.add_argument("--correctness-samples", type=int, default=2000)
    p.add_argument("--tiktoken-encoding", default="cl100k_base")
    p.add_argument("--tiktoken-num-threads", default="auto", help="int or auto")
    p.add_argument("--save-json", type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tf = _load_tokenflux()
    tiktoken = _load_tiktoken()
    hf_tokenizers = _load_hf_tokenizers()

    docs = _generate_docs(args.docs, args.min_phrases, args.max_phrases, args.seed)
    if not docs:
        raise RuntimeError("No docs generated.")
    chars = sum(len(doc) for doc in docs)

    tokenizer_path = args.tokenflux_tokenizer.resolve()
    if not tokenizer_path.exists():
        if not args.bootstrap_tokenizer:
            raise RuntimeError(f"Tokenizer not found: {tokenizer_path}")
        bootstrap_docs = docs[: max(1, min(len(docs), args.bootstrap_docs))]
        print(f"[setup] Training TokenFlux++ tokenizer -> {tokenizer_path}")
        _bootstrap_tokenizer(
            tf,
            tokenizer_path=tokenizer_path,
            docs=bootstrap_docs,
            trainer=args.trainer,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            min_pair_freq=args.min_pair_freq,
            threads=args.bootstrap_threads,
        )

    hf_tokenizer_path = args.tokenizers_tokenizer.resolve()
    if not hf_tokenizer_path.exists():
        if not args.bootstrap_tokenizer:
            raise RuntimeError(f"HuggingFace tokenizer not found: {hf_tokenizer_path}")
        bootstrap_docs = docs[: max(1, min(len(docs), args.bootstrap_docs))]
        print(f"[setup] Training HuggingFace tokenizers tokenizer -> {hf_tokenizer_path}")
        _train_hf_tokenizer(
            hf_tokenizers,
            tokenizer_path=hf_tokenizer_path,
            docs=bootstrap_docs,
            trainer_name=args.trainer,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            threads=args.bootstrap_threads,
        )

    tf_mode, tf_threads = _resolve_threads(args.tokenflux_threads, "--tokenflux-threads", allow_auto=True)
    hf_mode, hf_threads = _resolve_threads(args.tokenizers_threads, "--tokenizers-threads", allow_auto=True)
    train_mode, train_threads = _resolve_threads(args.train_threads, "--train-threads", allow_auto=True)
    tk_mode, tk_threads = _resolve_threads(str(args.tiktoken_num_threads), "--tiktoken-num-threads", allow_auto=True)
    thread_points = _parse_thread_points(args.thread_points)

    train_result: Optional[TimingResult] = None
    hf_train_result: Optional[TimingResult] = None
    train_docs_count = max(1, min(len(docs), args.train_docs))
    if args.benchmark_train:
        train_docs = docs[:train_docs_count]

        def run_train_once() -> None:
            out_path = ROOT / "artifacts" / f"benchmark_train_{uuid.uuid4().hex}.json"
            _bootstrap_tokenizer(
                tf,
                tokenizer_path=out_path,
                docs=train_docs,
                trainer=args.trainer,
                vocab_size=args.vocab_size,
                min_freq=args.min_freq,
                min_pair_freq=args.min_pair_freq,
                threads=train_threads,
            )
            try:
                out_path.unlink(missing_ok=True)
            except TypeError:  # Python < 3.8 compatibility guard
                if out_path.exists():
                    out_path.unlink()

        train_result = _run_timing(
            name=f"TokenFlux++ train (threads={train_threads})",
            runner=run_train_once,
            warmup=args.train_warmup,
            repeat=args.train_repeat,
        )
        if args.benchmark_tokenizers_train:
            def run_hf_train_once() -> None:
                out_path = ROOT / "artifacts" / f"benchmark_train_hf_{uuid.uuid4().hex}.json"
                _train_hf_tokenizer(
                    hf_tokenizers,
                    tokenizer_path=out_path,
                    docs=train_docs,
                    trainer_name=args.trainer,
                    vocab_size=args.vocab_size,
                    min_freq=args.min_freq,
                    threads=train_threads,
                )
                try:
                    out_path.unlink(missing_ok=True)
                except TypeError:
                    if out_path.exists():
                        out_path.unlink()

            hf_train_result = _run_timing(
                name=f"HuggingFace tokenizers train (threads={train_threads})",
                runner=run_hf_train_once,
                warmup=args.train_warmup,
                repeat=args.train_repeat,
            )

    enc = tiktoken.get_encoding(args.tiktoken_encoding)
    tf_runner = _make_tokenflux_runner(tf, tokenizer_path, docs, tf_threads)
    hf_runner = _make_hf_encode_runner(hf_tokenizers, hf_tokenizer_path, docs, hf_threads)
    tk_runner = _make_tiktoken_runner(enc, docs, tk_threads)

    tf_result = _run_benchmark(
        name=f"TokenFlux++ (threads={tf_threads})",
        runner=tf_runner,
        warmup=args.warmup,
        repeat=args.repeat,
        docs=len(docs),
        chars=chars,
    )
    tk_label = "OpenAI tiktoken (threads=auto)" if tk_mode == "auto" else f"OpenAI tiktoken (threads={tk_threads})"
    tk_result = _run_benchmark(
        name=tk_label,
        runner=tk_runner,
        warmup=args.warmup,
        repeat=args.repeat,
        docs=len(docs),
        chars=chars,
    )
    hf_result = _run_benchmark(
        name=f"HuggingFace tokenizers (threads={hf_threads})",
        runner=hf_runner,
        warmup=args.warmup,
        repeat=args.repeat,
        docs=len(docs),
        chars=chars,
    )

    print(f"Workload: docs={len(docs):,}, chars={chars:,}, avg_chars/doc={chars / len(docs):.1f}")
    if tf_mode == "auto":
        print(f"TokenFlux++ baseline threads: auto -> {tf_threads}")
    else:
        print(f"TokenFlux++ baseline threads: {tf_threads}")
    if hf_mode == "auto":
        print(f"HuggingFace tokenizers baseline threads: auto -> {hf_threads}")
    else:
        print(f"HuggingFace tokenizers baseline threads: {hf_threads}")
    if tk_mode == "auto":
        print(f"OpenAI tiktoken baseline threads: auto -> {tk_threads}")
    else:
        print(f"OpenAI tiktoken baseline threads: {tk_threads}")

    if train_result is not None:
        print("")
        print(
            f"Train benchmark: docs={train_docs_count:,}, "
            f"threads={'auto -> ' + str(train_threads) if train_mode == 'auto' else train_threads}"
        )
        train_rows = [
            [
                train_result.name,
                _fmtf(train_result.mean_latency),
                _fmtf(train_result.std_latency),
                _fmti(train_docs_count / train_result.mean_latency if train_result.mean_latency > 0 else 0.0),
            ]
        ]
        if hf_train_result is not None:
            train_rows.append(
                [
                    hf_train_result.name,
                    _fmtf(hf_train_result.mean_latency),
                    _fmtf(hf_train_result.std_latency),
                    _fmti(train_docs_count / hf_train_result.mean_latency if hf_train_result.mean_latency > 0 else 0.0),
                ]
            )
        _print_table(["Engine", "Mean latency (s)", "Std (s)", "Docs/s"], train_rows)

    print("")
    print("Encode benchmark")
    _print_table(
        ["Engine", "Mean latency (s)", "Std (s)", "Docs/s", "Chars/s", "Tokens/s", "Avg tokens/doc"],
        [
            [
                tf_result.name,
                _fmtf(tf_result.mean_latency),
                _fmtf(tf_result.std_latency),
                _fmti(tf_result.docs_per_sec),
                _fmti(tf_result.chars_per_sec),
                _fmti(tf_result.tokens_per_sec),
                _fmtf(tf_result.avg_tokens_per_doc),
            ],
            [
                tk_result.name,
                _fmtf(tk_result.mean_latency),
                _fmtf(tk_result.std_latency),
                _fmti(tk_result.docs_per_sec),
                _fmti(tk_result.chars_per_sec),
                _fmti(tk_result.tokens_per_sec),
                _fmtf(tk_result.avg_tokens_per_doc),
            ],
            [
                hf_result.name,
                _fmtf(hf_result.mean_latency),
                _fmtf(hf_result.std_latency),
                _fmti(hf_result.docs_per_sec),
                _fmti(hf_result.chars_per_sec),
                _fmti(hf_result.tokens_per_sec),
                _fmtf(hf_result.avg_tokens_per_doc),
            ],
        ],
    )

    speed_ratio = tk_result.mean_latency / tf_result.mean_latency
    if speed_ratio >= 1.0:
        print(f"Latency speedup: TokenFlux++ is {speed_ratio:.2f}x faster than OpenAI tiktoken.")
    else:
        print(f"Latency speedup: OpenAI tiktoken is {(1.0 / speed_ratio):.2f}x faster than TokenFlux++.")
    hf_speed_ratio = hf_result.mean_latency / tf_result.mean_latency
    if hf_speed_ratio >= 1.0:
        print(f"Latency speedup: TokenFlux++ is {hf_speed_ratio:.2f}x faster than HuggingFace tokenizers.")
    else:
        print(f"Latency speedup: HuggingFace tokenizers is {(1.0 / hf_speed_ratio):.2f}x faster than TokenFlux++.")

    decode_results: list[BenchmarkResult] = []
    tf_encoded_batch: list[list[int]] = []
    tk_encoded_batch: list[list[int]] = []
    hf_encoded_batch: list[list[int]] = []
    if args.benchmark_decode or args.correctness_samples > 0:
        tf_encode_for_decode = tf.Tokenizer(str(tokenizer_path))
        tf_encoded_batch = tf_encode_for_decode.encode_batch(docs, "", "", True)
        tk_encoded_batch = _encode_tiktoken_batch(enc, docs, tk_threads)
        hf_tok_for_decode = _create_hf_tokenizer(hf_tokenizers, hf_tokenizer_path)
        hf_encoded_batch = [item.ids for item in hf_tok_for_decode.encode_batch(docs)]

    if args.benchmark_decode:
        tf_decode_runner = _make_hf_decode_runner(hf_tokenizers, tokenizer_path, tf_encoded_batch, tf_threads)
        tf_decode_result = _run_benchmark(
            name=f"TokenFlux++ decode (threads={tf_threads})",
            runner=tf_decode_runner,
            warmup=args.warmup,
            repeat=args.repeat,
            docs=len(docs),
            chars=chars,
        )
        decode_results.append(tf_decode_result)

        tk_decode_tokens = sum(len(ids) for ids in tk_encoded_batch)

        def tk_decode_runner() -> int:
            _decode_tiktoken_batch(enc, tk_encoded_batch, tk_threads)
            return tk_decode_tokens

        tk_decode_result = _run_benchmark(
            name=tk_label.replace("tiktoken", "tiktoken decode"),
            runner=tk_decode_runner,
            warmup=args.warmup,
            repeat=args.repeat,
            docs=len(docs),
            chars=chars,
        )
        decode_results.append(tk_decode_result)

        hf_decode_runner = _make_hf_decode_runner(hf_tokenizers, hf_tokenizer_path, hf_encoded_batch, hf_threads)
        hf_decode_result = _run_benchmark(
            name=f"HuggingFace tokenizers decode (threads={hf_threads})",
            runner=hf_decode_runner,
            warmup=args.warmup,
            repeat=args.repeat,
            docs=len(docs),
            chars=chars,
        )
        decode_results.append(hf_decode_result)

        print("")
        print("Decode benchmark")
        _print_table(
            ["Engine", "Mean latency (s)", "Std (s)", "Docs/s", "Chars/s", "Tokens/s"],
            [
                [
                    item.name,
                    _fmtf(item.mean_latency),
                    _fmtf(item.std_latency),
                    _fmti(item.docs_per_sec),
                    _fmti(item.chars_per_sec),
                    _fmti(item.tokens_per_sec),
                ]
                for item in decode_results
            ],
        )

    correctness: dict[str, object] = {}
    check_n = max(0, min(args.correctness_samples, len(docs)))
    if check_n > 0:
        print("")
        print(f"Round-trip correctness check: samples={check_n:,}")
        hf_tok_tf = _create_hf_tokenizer(hf_tokenizers, tokenizer_path)
        hf_tok_hf = _create_hf_tokenizer(hf_tokenizers, hf_tokenizer_path)
        tf_tok_for_check = tf.Tokenizer(str(tokenizer_path))

        tf_ok = 0
        tk_ok = 0
        hf_ok = 0
        agree_tf_tk = 0
        agree_tf_hf = 0
        agree_tk_hf = 0

        for idx in range(check_n):
            tf_ids = tf_encoded_batch[idx]
            tk_ids = tk_encoded_batch[idx]
            hf_ids = hf_encoded_batch[idx]

            tf_text = hf_tok_tf.decode(tf_ids)
            tk_text = enc.decode(tk_ids)
            hf_text = hf_tok_hf.decode(hf_ids)

            if tf_tok_for_check.encode(tf_text, "", "", True) == tf_ids:
                tf_ok += 1
            if _encode_tiktoken_text(enc, tk_text) == tk_ids:
                tk_ok += 1
            if hf_tok_hf.encode(hf_text).ids == hf_ids:
                hf_ok += 1

            if tf_text == tk_text:
                agree_tf_tk += 1
            if tf_text == hf_text:
                agree_tf_hf += 1
            if tk_text == hf_text:
                agree_tk_hf += 1

        correctness = {
            "tokenflux": {"checked": check_n, "matched": tf_ok, "match_rate": tf_ok / check_n},
            "tiktoken": {"checked": check_n, "matched": tk_ok, "match_rate": tk_ok / check_n},
            "tokenizers": {"checked": check_n, "matched": hf_ok, "match_rate": hf_ok / check_n},
            "cross_agreement": {
                "tokenflux_vs_tiktoken": agree_tf_tk / check_n,
                "tokenflux_vs_tokenizers": agree_tf_hf / check_n,
                "tiktoken_vs_tokenizers": agree_tk_hf / check_n,
            },
        }

        rows = [
            ["TokenFlux++", str(check_n), str(tf_ok), _fmtf(100.0 * tf_ok / check_n) + "%"],
            ["OpenAI tiktoken", str(check_n), str(tk_ok), _fmtf(100.0 * tk_ok / check_n) + "%"],
            ["HuggingFace tokenizers", str(check_n), str(hf_ok), _fmtf(100.0 * hf_ok / check_n) + "%"],
        ]
        _print_table(["Engine", "Checked", "Matched", "Match rate"], rows)
        _print_table(
            ["Pair", "Text agreement rate"],
            [
                ["TokenFlux++ vs OpenAI tiktoken", _fmtf(100.0 * agree_tf_tk / check_n) + "%"],
                ["TokenFlux++ vs HuggingFace tokenizers", _fmtf(100.0 * agree_tf_hf / check_n) + "%"],
                ["OpenAI tiktoken vs HuggingFace tokenizers", _fmtf(100.0 * agree_tk_hf / check_n) + "%"],
            ],
        )

    print("")
    print(f"Thread points: {thread_points} (warmup={args.sweep_warmup}, repeat={args.sweep_repeat})")
    tk_by_thread: dict[int, BenchmarkResult] = {}
    hf_by_thread: dict[int, BenchmarkResult] = {}
    for th in thread_points:
        tk_by_thread[th] = _run_benchmark(
            name=f"OpenAI tiktoken (threads={th})",
            runner=_make_tiktoken_runner(enc, docs, th),
            warmup=args.sweep_warmup,
            repeat=args.sweep_repeat,
            docs=len(docs),
            chars=chars,
        )
        hf_by_thread[th] = _run_benchmark(
            name=f"HuggingFace tokenizers (threads={th})",
            runner=_make_hf_encode_runner(hf_tokenizers, hf_tokenizer_path, docs, th),
            warmup=args.sweep_warmup,
            repeat=args.sweep_repeat,
            docs=len(docs),
            chars=chars,
        )

    thread_rows: list[list[str]] = []
    thread_results: list[dict[str, object]] = []
    for th in thread_points:
        tf_point = _run_benchmark(
            name=f"TokenFlux++ (threads={th})",
            runner=_make_tokenflux_runner(tf, tokenizer_path, docs, th),
            warmup=args.sweep_warmup,
            repeat=args.sweep_repeat,
            docs=len(docs),
            chars=chars,
        )
        tk_point = tk_by_thread[th]
        hf_point = hf_by_thread[th]
        contenders = [
            ("TokenFlux++", tf_point.mean_latency),
            ("OpenAI tiktoken", tk_point.mean_latency),
            ("HuggingFace tokenizers", hf_point.mean_latency),
        ]
        faster = min(contenders, key=lambda item: item[1])[0]
        thread_rows.append(
            [
                str(th),
                _fmtf(tf_point.mean_latency),
                _fmtf(tk_point.mean_latency),
                _fmtf(hf_point.mean_latency),
                _fmti(tf_point.docs_per_sec),
                _fmti(tk_point.docs_per_sec),
                _fmti(hf_point.docs_per_sec),
                faster,
            ]
        )
        thread_results.append(
            {
                "threads": th,
                "tokenflux": {
                    "mean_latency_s": tf_point.mean_latency,
                    "std_latency_s": tf_point.std_latency,
                    "docs_per_sec": tf_point.docs_per_sec,
                    "chars_per_sec": tf_point.chars_per_sec,
                    "tokens_per_sec": tf_point.tokens_per_sec,
                },
                "tiktoken": {
                    "mean_latency_s": tk_point.mean_latency,
                    "std_latency_s": tk_point.std_latency,
                    "docs_per_sec": tk_point.docs_per_sec,
                    "chars_per_sec": tk_point.chars_per_sec,
                    "tokens_per_sec": tk_point.tokens_per_sec,
                },
                "tokenizers": {
                    "mean_latency_s": hf_point.mean_latency,
                    "std_latency_s": hf_point.std_latency,
                    "docs_per_sec": hf_point.docs_per_sec,
                    "chars_per_sec": hf_point.chars_per_sec,
                    "tokens_per_sec": hf_point.tokens_per_sec,
                },
            }
        )

    _print_table(
        [
            "Threads",
            "TokenFlux++ latency (s)",
            "tiktoken latency (s)",
            "tokenizers latency (s)",
            "TokenFlux++ docs/s",
            "tiktoken docs/s",
            "tokenizers docs/s",
            "Faster",
        ],
        thread_rows,
    )

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(
                {
                    "workload": {
                        "docs": len(docs),
                        "chars": chars,
                        "avg_chars_per_doc": chars / len(docs),
                        "tokenflux_tokenizer_path": str(tokenizer_path),
                        "tokenizers_tokenizer_path": str(hf_tokenizer_path),
                        "trainer": args.trainer,
                        "vocab_size": args.vocab_size,
                        "min_freq": args.min_freq,
                        "min_pair_freq": args.min_pair_freq,
                        "warmup": args.warmup,
                        "repeat": args.repeat,
                        "thread_points": thread_points,
                        "sweep_warmup": args.sweep_warmup,
                        "sweep_repeat": args.sweep_repeat,
                        "tokenflux_threads_mode": tf_mode,
                        "tokenflux_threads": tf_threads,
                        "tokenizers_threads_mode": hf_mode,
                        "tokenizers_threads": hf_threads,
                        "train_threads_mode": train_mode,
                        "train_threads": train_threads,
                        "train_docs": train_docs_count,
                        "train_warmup": args.train_warmup,
                        "train_repeat": args.train_repeat,
                        "tiktoken_threads_mode": tk_mode,
                        "tiktoken_num_threads": tk_threads,
                        "tiktoken_encoding": args.tiktoken_encoding,
                        "correctness_samples": check_n,
                    },
                    "results": [
                        {
                            "engine": tf_result.name,
                            "mean_latency_s": tf_result.mean_latency,
                            "std_latency_s": tf_result.std_latency,
                            "docs_per_sec": tf_result.docs_per_sec,
                            "chars_per_sec": tf_result.chars_per_sec,
                            "tokens_per_sec": tf_result.tokens_per_sec,
                            "avg_tokens_per_doc": tf_result.avg_tokens_per_doc,
                            "latencies_s": tf_result.latencies,
                        },
                        {
                            "engine": tk_result.name,
                            "mean_latency_s": tk_result.mean_latency,
                            "std_latency_s": tk_result.std_latency,
                            "docs_per_sec": tk_result.docs_per_sec,
                            "chars_per_sec": tk_result.chars_per_sec,
                            "tokens_per_sec": tk_result.tokens_per_sec,
                            "avg_tokens_per_doc": tk_result.avg_tokens_per_doc,
                            "latencies_s": tk_result.latencies,
                        },
                        {
                            "engine": hf_result.name,
                            "mean_latency_s": hf_result.mean_latency,
                            "std_latency_s": hf_result.std_latency,
                            "docs_per_sec": hf_result.docs_per_sec,
                            "chars_per_sec": hf_result.chars_per_sec,
                            "tokens_per_sec": hf_result.tokens_per_sec,
                            "avg_tokens_per_doc": hf_result.avg_tokens_per_doc,
                            "latencies_s": hf_result.latencies,
                        },
                    ],
                    "train_results": [
                        {
                            "engine": item.name,
                            "mean_latency_s": item.mean_latency,
                            "std_latency_s": item.std_latency,
                            "latencies_s": item.latencies,
                            "docs_per_sec": train_docs_count / item.mean_latency if item.mean_latency > 0 else 0.0,
                        }
                        for item in [x for x in [train_result, hf_train_result] if x is not None]
                    ],
                    "decode_results": [
                        {
                            "engine": item.name,
                            "mean_latency_s": item.mean_latency,
                            "std_latency_s": item.std_latency,
                            "docs_per_sec": item.docs_per_sec,
                            "chars_per_sec": item.chars_per_sec,
                            "tokens_per_sec": item.tokens_per_sec,
                            "latencies_s": item.latencies,
                        }
                        for item in decode_results
                    ],
                    "correctness": correctness,
                    "thread_point_results": thread_results,
                    "versions": {
                        "tokenflux": getattr(tf, "__version__", "unknown"),
                        "tiktoken": importlib.metadata.version("tiktoken"),
                        "python": sys.version.split()[0],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved JSON report: {args.save_json.resolve()}")


if __name__ == "__main__":
    main()
