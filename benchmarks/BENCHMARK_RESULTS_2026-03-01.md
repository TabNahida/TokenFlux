# Benchmark Results (2026-03-01)

Command:

```bash
python benchmarks/tokenfluxpp_vs_tiktoken_vs_tokenizer.py
```

## Workload

- docs: `200,000`
- chars: `130,129,434`
- avg chars/doc: `650.6`
- baseline threads:
  - TokenFlux++: `auto -> 32`
  - HuggingFace tokenizers: `auto -> 32`
  - OpenAI tiktoken: `auto -> 32`

## Train Benchmark

| Engine | Mean latency (s) | Std (s) | Docs/s |
|---|---:|---:|---:|
| TokenFlux++ train (threads=32) | 0.1020 | 0.0000 | 49,039 |
| HuggingFace tokenizers train (threads=32) | 0.1193 | 0.0000 | 41,914 |

## Encode Benchmark

| Engine | Mean latency (s) | Std (s) | Docs/s | Chars/s | Tokens/s | Avg tokens/doc |
|---|---:|---:|---:|---:|---:|---:|
| TokenFlux++ (threads=32) | 0.7789 | 0.0495 | 256,764 | 167,062,549 | 24,404,310 | 95.0458 |
| OpenAI tiktoken (threads=auto) | 3.3685 | 0.0307 | 59,373 | 38,630,762 | 6,621,511 | 111.5243 |
| HuggingFace tokenizers (threads=32) | 9.2588 | 1.6949 | 21,601 | 14,054,691 | 2,402,372 | 111.2153 |

Speedup:

- TokenFlux++ vs OpenAI tiktoken: **4.32x faster**
- TokenFlux++ vs HuggingFace tokenizers: **11.89x faster**

## Decode Benchmark

| Engine | Mean latency (s) | Std (s) | Docs/s | Chars/s | Tokens/s |
|---|---:|---:|---:|---:|---:|
| TokenFlux++ decode (threads=32) | 0.8422 | 0.0555 | 237,482 | 154,516,953 | 22,571,663 |
| OpenAI tiktoken decode (threads=auto) | 5.8339 | 0.4222 | 34,282 | 22,305,718 | 3,823,315 |
| HuggingFace tokenizers decode (threads=32) | 1.0327 | 0.0602 | 193,658 | 126,002,957 | 21,537,722 |

## Round-trip Correctness

Samples: `2,000`

| Engine | Checked | Matched | Match rate |
|---|---:|---:|---:|
| TokenFlux++ | 2000 | 2000 | 100.0000% |
| OpenAI tiktoken | 2000 | 2000 | 100.0000% |
| HuggingFace tokenizers | 2000 | 2000 | 100.0000% |

Cross-agreement (decoded text):

| Pair | Text agreement rate |
|---|---:|
| TokenFlux++ vs OpenAI tiktoken | 100.0000% |
| TokenFlux++ vs HuggingFace tokenizers | 100.0000% |
| OpenAI tiktoken vs HuggingFace tokenizers | 100.0000% |

## Thread Points

Config: `threads=[1,2,4,8,16]`, `warmup=1`, `repeat=2`

| Threads | TokenFlux++ latency (s) | tiktoken latency (s) | tokenizers latency (s) | TokenFlux++ docs/s | tiktoken docs/s | tokenizers docs/s | Faster |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.9700 | 10.2378 | 14.0205 | 67,339 | 19,535 | 14,265 | TokenFlux++ |
| 2 | 2.1469 | 8.6213 | 14.3224 | 93,159 | 23,198 | 13,964 | TokenFlux++ |
| 4 | 1.2912 | 6.6026 | 14.9484 | 154,900 | 30,291 | 13,379 | TokenFlux++ |
| 8 | 0.8943 | 6.1996 | 14.9420 | 223,631 | 32,260 | 13,385 | TokenFlux++ |
| 16 | 0.8135 | 6.0737 | 15.1337 | 245,839 | 32,929 | 13,216 | TokenFlux++ |

