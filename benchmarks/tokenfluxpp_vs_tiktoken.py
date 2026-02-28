from __future__ import annotations

import argparse
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
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = ROOT / "artifacts" / "benchmark_tokenizer.json"
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


def _make_tiktoken_runner(encoding, docs: list[str], threads: int) -> Callable[[], int]:
    def run() -> int:
        batch = _encode_tiktoken_batch(encoding, docs, threads)
        return sum(len(ids) for ids in batch)

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
    p.add_argument("--thread-points", default="1,2,4,8,16,32")
    p.add_argument("--sweep-warmup", type=int, default=1)
    p.add_argument("--sweep-repeat", type=int, default=2)
    p.add_argument("--tokenflux-tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--tokenflux-threads", default="auto", help="int or auto")
    p.add_argument("--trainer", default="byte_bpe")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--min-pair-freq", type=int, default=2)
    p.add_argument("--bootstrap-docs", type=int, default=5000)
    p.add_argument("--bootstrap-threads", type=int, default=4)
    p.add_argument("--bootstrap-tokenizer", dest="bootstrap_tokenizer", action="store_true")
    p.add_argument("--no-bootstrap-tokenizer", dest="bootstrap_tokenizer", action="store_false")
    p.set_defaults(bootstrap_tokenizer=True)
    p.add_argument("--tiktoken-encoding", default="cl100k_base")
    p.add_argument("--tiktoken-num-threads", type=int, default=0, help="<=0 means auto")
    p.add_argument("--save-json", type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tf = _load_tokenflux()
    tiktoken = _load_tiktoken()

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

    tf_mode, tf_threads = _resolve_threads(args.tokenflux_threads, "--tokenflux-threads", allow_auto=True)
    tk_threads = args.tiktoken_num_threads
    thread_points = _parse_thread_points(args.thread_points)

    enc = tiktoken.get_encoding(args.tiktoken_encoding)
    tf_runner = _make_tokenflux_runner(tf, tokenizer_path, docs, tf_threads)
    tk_runner = _make_tiktoken_runner(enc, docs, tk_threads)

    tf_result = _run_benchmark(
        name=f"TokenFlux++ (threads={tf_threads})",
        runner=tf_runner,
        warmup=args.warmup,
        repeat=args.repeat,
        docs=len(docs),
        chars=chars,
    )
    tk_label = "OpenAI tiktoken (threads=auto)" if tk_threads <= 0 else f"OpenAI tiktoken (threads={tk_threads})"
    tk_result = _run_benchmark(
        name=tk_label,
        runner=tk_runner,
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
    print("OpenAI tiktoken baseline threads: auto" if tk_threads <= 0 else f"OpenAI tiktoken baseline threads: {tk_threads}")

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
        ],
    )

    speed_ratio = tk_result.mean_latency / tf_result.mean_latency
    if speed_ratio >= 1.0:
        print(f"Latency speedup: TokenFlux++ is {speed_ratio:.2f}x faster than OpenAI tiktoken.")
    else:
        print(f"Latency speedup: OpenAI tiktoken is {(1.0 / speed_ratio):.2f}x faster than TokenFlux++.")

    print("")
    print(f"Thread points: {thread_points} (warmup={args.sweep_warmup}, repeat={args.sweep_repeat})")
    tk_by_thread: dict[int, BenchmarkResult] = {}
    for th in thread_points:
        tk_by_thread[th] = _run_benchmark(
            name=f"OpenAI tiktoken (threads={th})",
            runner=_make_tiktoken_runner(enc, docs, th),
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
        faster = "TokenFlux++" if tf_point.mean_latency <= tk_point.mean_latency else "OpenAI tiktoken"
        thread_rows.append(
            [
                str(th),
                _fmtf(tf_point.mean_latency),
                _fmtf(tk_point.mean_latency),
                _fmti(tf_point.docs_per_sec),
                _fmti(tk_point.docs_per_sec),
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
            }
        )

    _print_table(
        ["Threads", "TokenFlux++ latency (s)", "tiktoken latency (s)", "TokenFlux++ docs/s", "tiktoken docs/s", "Faster"],
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
                        "tokenizer_path": str(tokenizer_path),
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
                        "tiktoken_num_threads": tk_threads,
                        "tiktoken_encoding": args.tiktoken_encoding,
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
                    ],
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
