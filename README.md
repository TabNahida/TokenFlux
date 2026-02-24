# TokenFlux

TokenFlux 是一个基于 **C++20 + XMake** 的高性能 tokenizer / trainer 基础框架，目标是提供：

- 类似 HuggingFace Tokenizers 的核心能力（BPE 模型加载/保存、tokenize/detokenize）
- 面向海量文件的并行 pre-tokenize（输出二进制 shard）
- 可扩展的高性能 Byte Level BPE 训练器（chunk、TopK、并行）
- Python binding（pybind11，可选）
- CLI 入口用于训练和预处理

## 已实现（基础版）

### 1) Tokenizer

- 支持加载/保存 HuggingFace `tokenizer.json`（BPE 子集）
- 支持加载/保存 `vocab.json + merges.txt`
- 基础 tokenize / detokenize 流程

### 2) Trainer（Byte Level BPE）

- 先构建词频语料，再执行迭代 merge
- 并行 pair 统计（多线程）
- TopK 剪枝控制热点 pair，降低内存压力
- 并行执行 merge 更新

### 3) 海量数据预处理

- `PretokenizePipeline` 支持并行处理多个输入文件
- 输出 `.tokbin` shard：`[len(uint32_t)][token_ids(int32_t)...]`

### 4) 数据格式读取

- 已支持：`txt / json / jsonl / json.gz`
- 预留：`parquet`（可通过 `-o parquet=true` + Arrow 实现）

### 5) CLI

- `train_bpe`: 训练并导出 tokenizer
- `pretokenize`: 使用 tokenizer 进行并行预切分并存储

## 构建

```bash
xmake f -m release
xmake
```

可选 Python binding:

```bash
xmake f -m release -o python=true
xmake
```

## CLI 示例

训练 BPE：

```bash
./build/linux/x86_64/release/tokenflux_cli train_bpe 32000 out/model data/a.jsonl data/b.txt
```

预切分：

```bash
./build/linux/x86_64/release/tokenflux_cli pretokenize out/model.tokenizer.json out/shards data/a.jsonl data/b.json.gz
```

## 后续优化建议

- 真正落地 SIMD（UTF-8 扫描、pair 计数热点优化）
- 加入严格内存预算器（按 `memory_limit_mb` 实时裁剪哈希表）
- parquet 读取接入 Arrow Dataset 并做列式批处理
- 在线流式 tokenizer API（零拷贝 buffer）
- 增加 Unigram/WordPiece trainer
