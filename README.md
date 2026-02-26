# TokenFlux

TokenFlux is a high-performance C++ toolkit for tokenizer training and dataset pre-tokenization.

Latest release: **0.2.1**. See [RELEASE](https://github.com/TabNahida/TokenFlux/releases).

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
