# TokenFlux++

TokenFlux++ is a high-performance C++ toolkit for tokenizer training and dataset pre-tokenization.

Latest release: **0.3.2**. See [RELEASE](https://github.com/TabNahida/TokenFluxPlusPlus/releases).

## Binaries

- `TokenFluxTrain`: train a tokenizer from text corpora.
- `TokenFluxTokenize`: tokenize corpora into binary training shards.

Build:

```bash
xmake
```

Install from source with pip:

```bash
pip install .
```

Editable install for local development:

```bash
pip install -e .
```

`pip` packaging delegates native compilation to `xmake`, so a working `xmake` + C++ toolchain is still required.

Python binding build output:

- `build/windows/x64/release/tokenflux_cpp.pyd`

The binding exposes `TrainConfig`, `TokenizeArgs`, `train(...)`, and `tokenize(...)`.
It also exposes a runtime `Tokenizer` class that can load `tokenizer.json`, encode directly in Python, and return Torch tensors.

## Supported Training Backends

`TokenFluxTrain` supports:

- `byte_bpe`
- `bpe` (standard non-byte-level BPE)
- `wordpiece`
- `unigram`

### Example: Train

```bash
xmake run TokenFluxTrain \
  --data-list "data/inputs.list" \
  --trainer wordpiece \
  --vocab-size 50000 \
  --threads 8 \
  --output tokenizer.json \
  --vocab vocab.json
```

Key options:

- `--trainer {byte_bpe|bpe|wordpiece|unigram}`
- `--data-list` to read local paths, `file://...`, or `http(s)://...` entries from a local/remote list
- Default GPT-style special token set is now `<|endoftext|>`
- `--records-per-chunk` as the upper bound for per-task document batching
- `--threads` for multi-thread chunk processing
- `--chunk-dir` for resumable count chunks

## Performance Tuning

Training throughput depends heavily on runtime parameters and data shape. Use these as practical baselines.

### 1) CPU Utilization (`--threads`)

- Start with `--threads` near physical cores.
- If CPU is <70% and disk is not saturated, increase threads.
- If throughput drops after increasing threads, roll back by 2-4 threads.

Example:

```bash
xmake run TokenFluxTrain --threads 16 ...
```

### 2) Stream Chunking (`--records-per-chunk`)

- Streaming mode reads each file once, derives target batch size from file size, and uses `--records-per-chunk` as a hard cap.
- Smaller chunk cap: smoother balancing on highly skewed corpora, more queue overhead.
- Larger chunk cap: higher raw throughput on short-record corpora, less granular work stealing.
- Recommended range: `2000`-`10000` for JSONL-like corpora.

Example:

```bash
xmake run TokenFluxTrain --records-per-chunk 4000 ...
```

### 3) Queue Backpressure (`--queue-capacity`)

- Default `0` auto-derives from threads and is usually fine.
- If workers idle while reader is active, increase queue capacity.
- If memory spikes during ingestion, decrease queue capacity.

Example:

```bash
xmake run TokenFluxTrain --queue-capacity 128 ...
```

### 4) Input Cost Control (`--max-chars`)

- Long documents dominate preprocessing time.
- Set `--max-chars` to cap worst-case samples and stabilize per-doc latency.
- Typical values: `8000`-`30000` depending on corpus quality.

Example:

```bash
xmake run TokenFluxTrain --max-chars 12000 ...
```

### 5) Counting Pressure (`--top-k`, `--min-freq`, `--min-pair-freq`)

- Lower `--top-k` reduces per-chunk map size and merge cost.
- Increase `--min-freq` / `--min-pair-freq` to filter tail noise and speed pair-based trainers.
- For noisy web corpora, a faster preset is often:
  `--top-k 100000 --min-freq 3 --min-pair-freq 3`.

### 6) Memory-Bounded Mode (`--max-memory-mb`, `--pair-max-entries`)

- Use `--max-memory-mb` to enforce soft memory limits on counting/pair stats.
- Override `--pair-max-entries` only when you need tighter pair-map control.
- When memory is constrained, prefer lower `--top-k` and slightly higher freq thresholds.

### 7) Progress/ETA Stability (`--prescan`, `--progress-interval`)

- Default is now single-pass streaming with file-count progress only.
- Enable `--prescan` only if doc-total ETA matters more than reading each file once.
- `--progress-interval` too small can add logging overhead; `200`-`1000` ms is a good range.

### 8) Backend-Specific Notes

- `byte_bpe`, `bpe`, `wordpiece`: throughput is mainly dominated by counting and pair stats.
- `unigram`: extra EM iterations are expensive; reduce `--unigram-iters` and seed size if speed-first.

Speed-first unigram example:

```bash
xmake run TokenFluxTrain \
  --trainer unigram \
  --unigram-iters 3 \
  --unigram-seed-mult 3 \
  --unigram-prune-ratio 0.8 ...
```

## Benchmark: TokenFlux++ vs OpenAI tiktoken

Use the benchmark script to compare batch encoding latency and throughput:

```bash
python benchmarks/tokenfluxpp_vs_tiktoken.py \
  --tokenflux-tokenizer artifacts/benchmark_tokenizer.json \
  --docs 20000 \
  --warmup 1 \
  --repeat 5 \
  --save-json artifacts/benchmark_report.json
```

Notes:

- Install comparison dependency: `python -m pip install tiktoken`
- If `artifacts/benchmark_tokenizer.json` does not exist, the script auto-trains a temporary TokenFlux++ byte-level BPE tokenizer from benchmark text (`--bootstrap-tokenizer` is enabled by default).
- You can benchmark your own corpus with `--input-txt` (one doc per line) or `--input-jsonl --text-field text`.
- `--tiktoken-encoding` defaults to `cl100k_base`; change it to match your target OpenAI tokenizer setup.
- Output includes mean latency, standard deviation, docs/s, chars/s, and tokens/s for both engines, plus a JSON report for charts.

## Pre-Tokenization (Sharding)

`TokenFluxTokenize` loads `tokenizer.json` and automatically supports model types:

- `BPE`
- `WordPiece`
- `Unigram`

It also respects tokenizer pre-tokenization style (`ByteLevel` / `WhitespaceSplit`) used by exported models.

### Example: Tokenize

```bash
xmake run TokenFluxTokenize \
  --data-list "data/inputs.list" \
  --tokenizer tokenizer.json \
  --out-dir data/tokens \
  --threads 8 \
  --max-tokens-per-shard 50000000
```

Output layout:

- `out-dir/shards/train_XXXXXX.bin`
- `out-dir/cache/completed.list`
- `out-dir/meta.json`

Behavior notes:

- Input can come from `--data-glob` or `--data-list`; list files can be local paths or remote `http(s)` URLs.
- List entries can be local paths, `file://` URLs, or remote `http(s)` files. Remote inputs are fetched through `cpp-httplib` and cached in C++ before parsing.
- Token output is written directly to `shards/` during tokenization (no duplicated part binaries).
- `--resume` reuses `cache/completed.list` to skip completed source files.
- Default progress is file-based streaming progress; `--prescan` is optional.

### Python Binding Example

```python
import tokenflux as tf

cfg = tf.TrainConfig()
cfg.trainer = tf.TrainerKind.bpe
cfg.vocab_size = 32000
tf.train(cfg, [r"data\train_000.jsonl", r"data\train_001.jsonl"])

args = tf.TokenizeArgs()
args.tokenizer_path = r"tokenizer.json"
args.out_dir = r"data\tokens"
tf.tokenize(args, [r"data\train_000.jsonl", r"data\train_001.jsonl"])

tok = tf.Tokenizer(r"tokenizer.json")
ids = tok.encode("hello world")
batch = tok.encode_batch_to_torch(["hello world", "token flux"], pad_id=0)
streamed = tok.tokenize_inputs_to_torch([r"data\train_000.jsonl"], text_field="text")
```

## Project Structure

- `tokenizer/TokenFluxTrain.cpp`: unified train entry.
- `tokenizer/TokenFluxTokenize.cpp`: tokenize CLI entry.
- `tokenizer/input_source.*`: local/remote input list resolution and remote cache materialization.
- `tokenizer/tokenize_common.*`: tokenize shared args/types/helpers.
- `tokenizer/tokenize_tokenizer.*`: tokenizer.json parsing + runtime encoder.
- `tokenizer/tokenize_pipeline.*`: shard writing, resume state, and tokenize pipeline.
- `tokenizer/train_frontend.*`: CLI/env parsing for training.
- `tokenizer/train_pipeline.*`: train runtime entry used by CLI and Python binding.
- `tokenizer/train_io.*`: chunked concurrent reading/writing + progress.
- `tokenizer/tokenflux_pybind.cpp`: Python binding module.
- `tokenizer/train_backend_*.cpp`: per-backend training implementations.
- `tokenizer/train_backend_common.*`: shared training helpers, including BPE pair-merge training.
- `tokenizer/trainers.*`: backend dispatch + tokenizer export.

## Notes

- `.env` is supported for common defaults (for example `DATA_PATH`, `DATA_LIST`, `VOCAB_SIZE`, `THREADS`).
- `TokenFluxTrain` and `TokenFluxTokenize` both support resumable workflows.
