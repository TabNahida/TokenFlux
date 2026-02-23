#include <cassert>
#include <vector>

#include "tokenflux/tokenizer.hpp"
#include "tokenflux/trainer.hpp"

int main() {
  using namespace tokenflux;

  std::vector<std::string> corpus = {
      "aaaaab",
      "aaaabb",
      "bbbbcc",
  };

  TrainerOptions opts;
  opts.vocab_size = 260;
  opts.min_frequency = 1;
  ByteLevelBPETrainer trainer(opts);
  auto trained = trainer.TrainFromLines(corpus);
  assert(trained.vocab.size() >= 256);

  ByteLevelBPETokenizer tokenizer(trained.vocab, trained.merges, 0);
  auto ids = tokenizer.Encode("aaaabb");
  assert(!ids.empty());
  auto text = tokenizer.Decode(ids);
  assert(!text.empty());

  return 0;
}

