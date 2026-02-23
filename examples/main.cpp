#include <iostream>
#include <vector>

#include "tokenflux/formats.hpp"
#include "tokenflux/tokenizer.hpp"
#include "tokenflux/trainer.hpp"

int main() {
  using namespace tokenflux;

  std::vector<std::string> corpus = {
      "TokenFlux focuses on high throughput tokenization.",
      "Byte level BPE merges frequent byte pairs.",
      "XMake based C++ library with Python bindings.",
  };

  TrainerOptions opts;
  opts.vocab_size = 300;
  opts.min_frequency = 1;
  opts.num_threads = 4;
  ByteLevelBPETrainer trainer(opts);

  auto result = trainer.TrainFromLines(corpus);
  SaveAsGPT2(result, "vocab.txt", "merges.txt");

  auto tokenizer = ByteLevelBPETokenizer(result.vocab, result.merges, 0);
  auto ids = tokenizer.Encode("TokenFlux streaming encode demo");
  std::cout << "Encoded IDs:";
  for (auto id : ids) {
    std::cout << ' ' << id;
  }
  std::cout << "\nDecoded: " << tokenizer.Decode(ids) << '\n';
  return 0;
}

