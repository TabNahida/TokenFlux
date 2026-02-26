# TokenFlux

TokenFlux is a high-performance C++ toolkit for tokenizer training and dataset pre-tokenization.

Latest release: **0.2.2**. See [RELEASE](https://github.com/TabNahida/TokenFlux/releases).

## Binaries

- `TokenFluxTrain`: train a tokenizer from text corpora.
- `TokenFluxTokenize`: tokenize corpora into binary training shards.

Build:

```bash
xmake
```

## Supported Training Backends

`TokenFluxTrain` supports:

- `byte_bpe`
- `bpe` (standard non-byte-level BPE)
- `wordpiece`
- `unigram`

### Example: Train

```bash
xmake run TokenFluxTrain \
  --data "data/*.jsonl" \
  --trainer wordpiece \
  --vocab-size 50000 \
  --records-per-chunk 5000 \
  --threads 8 \
  --output tokenizer.json \
  --vocab vocab.json
```

Key options:

- `--trainer {byte_bpe|bpe|wordpiece|unigram}`
- `--records-per-chunk` for fine-grained chunking/progress on large files
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

### 2) Chunk Granularity (`--records-per-chunk`)

- Smaller chunk size: smoother progress updates, more scheduling overhead.
- Larger chunk size: higher raw throughput, rougher ETA/progress and larger per-task latency.
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

- Keep `--prescan` enabled (default) for stable `docs total` and more reliable ETA.
- Use `--no-prescan` only when startup latency matters more than ETA quality.
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

## Pre-Tokenization (Sharding)

`TokenFluxTokenize` loads `tokenizer.json` and automatically supports model types:

- `BPE`
- `WordPiece`
- `Unigram`

It also respects tokenizer pre-tokenization style (`ByteLevel` / `WhitespaceSplit`) used by exported models.

### Example: Tokenize

```bash
xmake run TokenFluxTokenize \
  --data-glob "data/*.jsonl" \
  --tokenizer tokenizer.json \
  --out-dir data/tokens \
  --threads 8 \
  --max-tokens-per-shard 50000000
```

Output layout:

- `out-dir/shards/train_XXXXXX.bin`
- `out-dir/cache/parts/part_XXXXXX.bin`
- `out-dir/meta.json`

## Project Structure

- `tokenizer/TokenFluxTrain.cpp`: unified train entry.
- `tokenizer/TokenFluxTokenize.cpp`: unified tokenize entry.
- `tokenizer/train_frontend.*`: CLI/env parsing for training.
- `tokenizer/train_io.*`: chunked concurrent reading/writing + progress.
- `tokenizer/train_backend_*.cpp`: per-backend training implementations.
- `tokenizer/trainers.*`: backend dispatch + tokenizer export.

## Notes

- `.env` is supported for common defaults (for example `DATA_PATH`, `VOCAB_SIZE`, `THREADS`).
- `TokenFluxTrain` and `TokenFluxTokenize` both support resumable workflows.
