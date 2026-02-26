#include "tokenflux_lib.hpp"
#include "train_frontend.hpp"
#include "train_pipeline.hpp"

#include <iostream>

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Config cfg;

    cfg.env_path = detect_env_path_arg(argc, argv, cfg.env_path);
    auto env = read_env_file(cfg.env_path);
    apply_env_overrides(cfg, env);

    std::string parse_err;
    bool show_help = false;
    if (!parse_train_args(argc, argv, cfg, parse_err, show_help))
    {
        if (show_help)
        {
            print_train_usage();
            return 0;
        }
        std::cerr << parse_err << "\n";
        print_train_usage();
        return 1;
    }

    return run_train(cfg);
}
