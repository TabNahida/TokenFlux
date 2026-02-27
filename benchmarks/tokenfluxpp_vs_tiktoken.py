from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = ROOT / "artifacts" / "benchmark_tokenizer.json"
SYNTHETIC_SNIPPETS = [
    "TokenFlux++ focuses on low-latency tokenizer inference for production data pipelines.",
    "The runtime is implemented in C++ and exposed to Python through pybind bindings.",
    "Stable throughput under mixed document sizes is critical for pretraining workloads.",
    "Work stealing and bounded queues keep worker utilization high on multi-core CPUs.",
    "Pre-tokenization quality depends on tokenizer.json compatibility and text normalization.",
    "Batch encoding should remain deterministic across benchmark rounds for fair comparison.",
    "Throughput in tokens per second and latency per batch are both useful metrics.",
    "Benchmark methodology should include warmup iterations to reduce startup noise.",
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
    for candidate in (ROOT / "build").rglob("tokenflux_cpp.pyd"):
        build_dir = str(candidate.parent)
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        return
    for candidate in (ROOT / "build").rglob("tokenflux_cpp.so"):
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
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Failed to import tokenflux. Build/install tokenflux_cpp first "
            "(example: `python -m pip install -e . --no-build-isolation`)."
        ) from exc


def _load_tiktoken():
    if importlib.util.find_spec("tiktoken") is None:
        raise RuntimeError(
            "Missing dependency: tiktoken. Install it with `python -m pip install tiktoken`."
        )
    return importlib.import_module("tiktoken")


def _load_docs_from_txt(path: Path) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                docs.append(text)
    return docs


def _load_docs_from_jsonl(path: Path, text_field: str) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}") from exc
            text = obj.get(text_field, "")
            if isinstance(text, str) and text:
                docs.append(text)
    return docs


def _generate_synthetic_docs(num_docs: int, min_phrases: int, max_phrases: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    docs: list[str] = []
    for idx in range(num_docs):
        phrase_count = rng.randint(min_phrases, max_phrases)
        phrases = [SYNTHETIC_SNIPPETS[rng.randrange(len(SYNTHETIC_SNIPPETS))] for _ in range(phrase_count)]
        phrases.append(f"doc_id={idx}")
        docs.append(" ".join(phrases))
    return docs


def _bootstrap_tokenizer(tf, tokenizer_path: Path, docs: list[str], vocab_size: int, threads: int) -> None:
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tokenfluxpp_bench_", dir=artifacts_dir) as tmp:
        tmpdir = Path(tmp)
        sample = tmpdir / "bootstrap.jsonl"
        with sample.open("w", encoding="utf-8") as f:
            for text in docs:
                f.write(json.dumps({"text": text}))
                f.write("\n")

        cfg = tf.TrainConfig()
        cfg.trainer = tf.TrainerKind.byte_bpe
        cfg.vocab_size = vocab_size
        cfg.min_freq = 2
        cfg.min_pair_freq = 2
        cfg.threads = max(1, threads)
        cfg.resume = False
        cfg.chunk_dir = str(tmpdir / "chunks")
        cfg.output_json = str(tokenizer_path)
        cfg.output_vocab = str(tmpdir / "vocab.json")
        cfg.output_merges = str(tmpdir / "merges.txt")
        tf.train(cfg, [str(sample)])


def _encode_tiktoken_batch(encoding, docs: list[str], num_threads: int) -> list[list[int]]:
    kwargs = {}
    if num_threads > 0:
        kwargs["num_threads"] = num_threads
    if hasattr(encoding, "encode_ordinary_batch"):
        try:
            return encoding.encode_ordinary_batch(docs, **kwargs)
        except TypeError:
            return encoding.encode_ordinary_batch(docs)
    try:
        return encoding.encode_batch(docs, disallowed_special=(), **kwargs)
    except TypeError:
        return encoding.encode_batch(docs, disallowed_special=())


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
        start = perf_counter()
        token_count = runner()
        elapsed = perf_counter() - start
        latencies.append(elapsed)
        token_counts.append(token_count)
    return BenchmarkResult(name=name, latencies=latencies, token_counts=token_counts, docs=docs, chars=chars)


def _fmt_float(value: float) -> str:
    return f"{value:,.4f}"


def _fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TokenFlux++ against OpenAI tiktoken for batch encoding throughput."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-txt", type=Path, help="Plain text file. One document per line.")
    input_group.add_argument(
        "--input-jsonl",
        type=Path,
        help="JSONL input file. Use --text-field to choose the text key (default: text).",
    )
    parser.add_argument("--text-field", default="text", help="Text field name for --input-jsonl.")
    parser.add_argument("--docs", type=int, default=20000, help="Synthetic docs count or max docs from input file.")
    parser.add_argument("--min-phrases", type=int, default=3, help="Min snippets per synthetic document.")
    parser.add_argument("--max-phrases", type=int, default=12, help="Max snippets per synthetic document.")
    parser.add_argument("--seed", type=int, default=20260227, help="Synthetic workload random seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds (not measured).")
    parser.add_argument("--repeat", type=int, default=5, help="Measured rounds.")
    parser.add_argument(
        "--tokenflux-tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Tokenizer JSON path for TokenFlux++ (default: {DEFAULT_TOKENIZER_PATH}).",
    )
    parser.add_argument(
        "--bootstrap-tokenizer",
        dest="bootstrap_tokenizer",
        action="store_true",
        help="Train a byte-level BPE tokenizer when --tokenflux-tokenizer does not exist.",
    )
    parser.add_argument(
        "--no-bootstrap-tokenizer",
        dest="bootstrap_tokenizer",
        action="store_false",
        help="Do not auto-train tokenizer when --tokenflux-tokenizer is missing.",
    )
    parser.set_defaults(bootstrap_tokenizer=True)
    parser.add_argument("--bootstrap-docs", type=int, default=5000, help="Docs used for tokenizer bootstrap training.")
    parser.add_argument("--bootstrap-vocab-size", type=int, default=16000, help="Vocab size for bootstrap training.")
    parser.add_argument("--bootstrap-threads", type=int, default=4, help="Threads for bootstrap training.")
    parser.add_argument("--tiktoken-encoding", default="cl100k_base", help="OpenAI tiktoken encoding name.")
    parser.add_argument(
        "--tiktoken-num-threads",
        type=int,
        default=0,
        help="tiktoken batch encode threads (<=0 means use tiktoken default).",
    )
    parser.add_argument("--save-json", type=Path, help="Optional path to write benchmark results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_phrases < args.min_phrases:
        raise ValueError("--max-phrases must be >= --min-phrases")

    if args.input_txt:
        docs = _load_docs_from_txt(args.input_txt)
    elif args.input_jsonl:
        docs = _load_docs_from_jsonl(args.input_jsonl, args.text_field)
    else:
        docs = _generate_synthetic_docs(args.docs, args.min_phrases, args.max_phrases, args.seed)

    if args.docs > 0:
        docs = docs[: args.docs]
    if not docs:
        raise RuntimeError("No documents available for benchmarking.")

    tf = _load_tokenflux()
    tiktoken = _load_tiktoken()

    tokenizer_path = args.tokenflux_tokenizer.resolve()
    if not tokenizer_path.exists():
        if not args.bootstrap_tokenizer:
            raise RuntimeError(
                f"Tokenizer not found: {tokenizer_path}. "
                "Provide --tokenflux-tokenizer or enable --bootstrap-tokenizer."
            )
        bootstrap_docs = docs[: max(1, min(len(docs), args.bootstrap_docs))]
        print(f"[setup] Training bootstrap TokenFlux++ tokenizer -> {tokenizer_path}")
        _bootstrap_tokenizer(
            tf,
            tokenizer_path=tokenizer_path,
            docs=bootstrap_docs,
            vocab_size=args.bootstrap_vocab_size,
            threads=args.bootstrap_threads,
        )

    tokenflux_tokenizer = tf.Tokenizer(str(tokenizer_path))
    tiktoken_encoding = tiktoken.get_encoding(args.tiktoken_encoding)

    total_chars = sum(len(doc) for doc in docs)

    def run_tokenflux() -> int:
        batch = tokenflux_tokenizer.encode_batch(docs, reset_cache=True)
        return sum(len(ids) for ids in batch)

    def run_tiktoken() -> int:
        batch = _encode_tiktoken_batch(tiktoken_encoding, docs, args.tiktoken_num_threads)
        return sum(len(ids) for ids in batch)

    tokenflux_result = _run_benchmark(
        name="TokenFlux++",
        runner=run_tokenflux,
        warmup=args.warmup,
        repeat=args.repeat,
        docs=len(docs),
        chars=total_chars,
    )
    tiktoken_result = _run_benchmark(
        name="OpenAI tiktoken",
        runner=run_tiktoken,
        warmup=args.warmup,
        repeat=args.repeat,
        docs=len(docs),
        chars=total_chars,
    )

    print(
        f"Workload: docs={len(docs):,}, chars={total_chars:,}, "
        f"avg_chars/doc={total_chars / len(docs):.1f}"
    )
    print(f"TokenFlux++ tokenizer: {tokenizer_path}")
    print(f"OpenAI tiktoken encoding: {args.tiktoken_encoding}")

    print("| Engine | Mean latency (s) | Std (s) | Docs/s | Chars/s | Tokens/s | Avg tokens/doc |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in (tokenflux_result, tiktoken_result):
        print(
            f"| {result.name} | {_fmt_float(result.mean_latency)} | {_fmt_float(result.std_latency)} | "
            f"{_fmt_int(result.docs_per_sec)} | {_fmt_int(result.chars_per_sec)} | "
            f"{_fmt_int(result.tokens_per_sec)} | {_fmt_float(result.avg_tokens_per_doc)} |"
        )

    tf_v = getattr(tf, "__version__", "unknown")
    tk_v = importlib.metadata.version("tiktoken")
    print(f"Versions: tokenflux={tf_v}, tiktoken={tk_v}, python={sys.version.split()[0]}")

    speed_ratio = tiktoken_result.mean_latency / tokenflux_result.mean_latency
    if speed_ratio >= 1.0:
        print(f"Latency speedup: TokenFlux++ is {speed_ratio:.2f}x faster than OpenAI tiktoken.")
    else:
        print(f"Latency speedup: OpenAI tiktoken is {(1.0 / speed_ratio):.2f}x faster than TokenFlux++.")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(
                {
                    "workload": {
                        "docs": len(docs),
                        "chars": total_chars,
                        "avg_chars_per_doc": total_chars / len(docs),
                        "tokenflux_tokenizer": str(tokenizer_path),
                        "tiktoken_encoding": args.tiktoken_encoding,
                        "warmup": args.warmup,
                        "repeat": args.repeat,
                    },
                    "results": [
                        {
                            "engine": tokenflux_result.name,
                            "mean_latency_s": tokenflux_result.mean_latency,
                            "std_latency_s": tokenflux_result.std_latency,
                            "docs_per_sec": tokenflux_result.docs_per_sec,
                            "chars_per_sec": tokenflux_result.chars_per_sec,
                            "tokens_per_sec": tokenflux_result.tokens_per_sec,
                            "avg_tokens_per_doc": tokenflux_result.avg_tokens_per_doc,
                            "mean_token_count": tokenflux_result.mean_tokens,
                            "latencies_s": tokenflux_result.latencies,
                        },
                        {
                            "engine": tiktoken_result.name,
                            "mean_latency_s": tiktoken_result.mean_latency,
                            "std_latency_s": tiktoken_result.std_latency,
                            "docs_per_sec": tiktoken_result.docs_per_sec,
                            "chars_per_sec": tiktoken_result.chars_per_sec,
                            "tokens_per_sec": tiktoken_result.tokens_per_sec,
                            "avg_tokens_per_doc": tiktoken_result.avg_tokens_per_doc,
                            "mean_token_count": tiktoken_result.mean_tokens,
                            "latencies_s": tiktoken_result.latencies,
                        },
                    ],
                    "versions": {
                        "tokenflux": tf_v,
                        "tiktoken": tk_v,
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
