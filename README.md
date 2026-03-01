# TokenFlux++

`TokenFlux++` is a fast tokenizer toolkit (C++ core + Python bindings) for:

- Training tokenizer models (`byte_bpe`, `bpe`, `wordpiece`, `unigram`)
- High-throughput encoding and dataset pre-tokenization

Latest release: **0.3.2**  
Releases: https://github.com/TabNahida/TokenFluxPlusPlus/releases

## Install

```bash
pip install .
```

Editable install:

```bash
pip install -e .
```

`xmake` and a C++ toolchain are required because native modules are built from source.

## Quickstart (Python)

```python
import tokenflux as tf

# train
cfg = tf.TrainConfig()
cfg.trainer = tf.TrainerKind.byte_bpe
cfg.vocab_size = 16000
cfg.output_json = "tokenizer.json"
cfg.output_vocab = "vocab.json"
cfg.output_merges = "merges.txt"
tf.train(cfg, ["data/train.jsonl"])

# encode
tok = tf.Tokenizer("tokenizer.json")
ids = tok.encode("hello TokenFlux++")
print(ids[:10], len(ids))
```

## Performance

Benchmark command:

```bash
python benchmarks/tokenfluxpp_vs_tiktoken.py
```

The benchmark uses built-in synthetic workload (`docs=200000`) and compares:

- Train latency (TokenFlux++ only)
- Baseline: `TokenFlux++ auto` vs `OpenAI tiktoken auto`
- Decode latency (TokenFlux++ vs OpenAI tiktoken)
- Round-trip correctness: `encode -> decode -> re-encode`
- Thread points: `1,2,4,8,16`

Install compare dependency:

```bash
python -m pip install tiktoken
python -m pip install tokenizers
```

### Latest benchmark result

Workload:

- docs: `200,000`
- chars: `130,129,434`
- avg chars/doc: `650.6`

Baseline (`auto` vs `auto`):

| Engine | Mean latency (s) | Std (s) | Docs/s | Chars/s | Tokens/s | Avg tokens/doc |
|---|---:|---:|---:|---:|---:|---:|
| TokenFlux++ (threads=32) | 0.7761 | 0.0910 | 257,696 | 167,669,194 | 24,492,928 | 95.0458 |
| OpenAI tiktoken (threads=auto) | 3.5128 | 0.1266 | 56,935 | 37,044,604 | 6,349,635 | 111.5243 |

Latency speedup: **TokenFlux++ is 4.53x faster than OpenAI tiktoken**.

Thread points:

| Threads | TokenFlux++ latency (s) | tiktoken latency (s) | TokenFlux++ docs/s | tiktoken docs/s | Faster |
|---:|---:|---:|---:|---:|---|
| 1 | 2.9136 | 7.4206 | 68,643 | 26,952 | TokenFlux++ |
| 2 | 2.0457 | 4.8981 | 97,765 | 40,832 | TokenFlux++ |
| 4 | 1.3322 | 3.3811 | 150,124 | 59,152 | TokenFlux++ |
| 8 | 0.7972 | 3.5846 | 250,872 | 55,794 | TokenFlux++ |
| 16 | 0.7364 | 3.4150 | 271,596 | 58,566 | TokenFlux++ |

## CLI

Train:

```bash
xmake run TokenFluxTrain \
  --data-list "data/inputs.list" \
  --trainer byte_bpe \
  --vocab-size 16000 \
  --threads 8 \
  --output tokenizer.json \
  --vocab vocab.json
```

Tokenize:

```bash
xmake run TokenFluxTokenize \
  --data-list "data/inputs.list" \
  --tokenizer tokenizer.json \
  --out-dir data/tokens \
  --threads 8 \
  --max-tokens-per-shard 50000000
```

## Build

```bash
xmake
```

Python extension output is typically under `build/.../tokenflux_cpp.pyd` (Windows) or `build/.../tokenflux_cpp.so` (Linux/macOS).

## Notes

- To pick up binding updates (for example encode threading changes), rebuild:

```bash
xmake f -y -m release --pybind=y
xmake build -y tokenflux_cpp
```

- `.env` defaults are supported for CLI workflows.
