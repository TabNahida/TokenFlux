from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

threads = [1, 2, 4, 8, 16]
values = {
    "TokenFlux++": [67339, 93159, 154900, 223631, 245839],
    "OpenAI tiktoken": [19535, 23198, 30291, 32260, 32929],
    "HF tokenizers": [14265, 13964, 13379, 13385, 13216],
}

index = np.arange(len(threads))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
ax.bar(index - bar_width, values["TokenFlux++"], bar_width, label="TokenFlux++", color="#1f77b4")
ax.bar(index, values["OpenAI tiktoken"], bar_width, label="OpenAI tiktoken", color="#ff7f0e")
ax.bar(index + bar_width, values["HF tokenizers"], bar_width, label="HF tokenizers", color="#2ca02c")

ax.set_xlabel("Threads")
ax.set_ylabel("Docs/s")
ax.set_title("Encode Throughput (docs/s) by threads")
ax.set_xticks(index)
ax.set_xticklabels([str(t) for t in threads])
ax.set_ylim(0, 280000)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
out = "benchmarks/thread_throughput.png"
fig.savefig(out)
print(out)
