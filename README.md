# TokenFlux

高性能 C++20 + XMake Tokenizer/Trainer 原型，目标是提供：

- 类似 HF Tokenizers 的格式兼容（GPT2 `vocab.txt + merges.txt`，以及 `tokenizer.json` 导出）
- 面向海量语料的并行 pre-tokenize 并持久化（二进制 token 流 + 索引）
- 面向 Python 的高效 binding（pybind11）
- 训练侧先支持 Byte-Level BPE，并实现针对性优化框架（chunk、TopK、并行）

## Build

```bash
xmake f -m release
xmake
xmake run tokenflux_cli
```

构建 Python 模块：

```bash
xmake f --python=y
xmake
```

## 当前实现要点

1. **Tokenizer**
   - `ByteLevelBPETokenizer`：支持基于 merges 的贪心合并编码与解码。
   - `StreamingEncoder`：流式 push + flush，适合 Python binding 场景的增量处理。

2. **Pretokenize**
   - `PretokenizeEngine` 并行编码多文件，并写出：
     - `*.bin`：连续 token id 序列
     - `*.idx`：`(offset, length)` 索引

3. **Trainer (Byte-Level BPE)**
   - 语料预编码为 byte ids。
   - 分片并行统计 pair 频次。
   - 每轮使用 TopK 裁剪候选，降低峰值内存与排序代价。
   - 受 `max_memory_bytes` 约束控制训练数据读取。

## 下一步性能增强建议

- pair 统计热路径用 SIMD（AVX2/AVX-512）+ 无锁分桶聚合
- 使用 mmap + NUMA 感知分配提升海量语料吞吐
- 将 merge 更新切为 block 并并行 prefix-scan，减少重分配
- 提供 Rust/Python wheels，补齐 HF tokenizer.json 的完整字段兼容

