#include "tokenflux/byte_level_bpe_trainer.hpp"
#include "tokenflux/pretokenizer.hpp"

#include <iostream>

using namespace tokenflux;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  tokenflux_cli train_bpe <vocab_size> <output_prefix> <files...>\n"
                  << "  tokenflux_cli pretokenize <tokenizer.json> <output_dir> <files...>\n";
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "train_bpe") {
        if (argc < 5) return 1;
        TrainerOptions opts;
        opts.vocab_size = static_cast<size_t>(std::stoul(argv[2]));
        ByteLevelBPETrainer trainer(opts);
        std::vector<std::string> files;
        for (int i = 4; i < argc; ++i) files.emplace_back(argv[i]);
        auto tok = trainer.train(files);
        std::string prefix = argv[3];
        tok.save_hf_tokenizer_json(prefix + ".tokenizer.json");
        tok.save_bpe_files(prefix + ".vocab.json", prefix + ".merges.txt");
        std::cout << "trained vocab=" << tok.vocab().size() << "\n";
        return 0;
    }

    if (cmd == "pretokenize") {
        if (argc < 5) return 1;
        Tokenizer tok;
        if (!tok.load_hf_tokenizer_json(argv[2])) {
            std::cerr << "failed to load tokenizer\n";
            return 2;
        }
        PretokenizeOptions popts;
        popts.output_dir = argv[3];
        std::vector<std::string> files;
        for (int i = 4; i < argc; ++i) files.emplace_back(argv[i]);
        PretokenizePipeline pipeline(tok);
        if (!pipeline.run(files, popts)) {
            std::cerr << "pretokenize failed\n";
            return 3;
        }
        std::cout << "pretokenize done\n";
        return 0;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    return 1;
}
