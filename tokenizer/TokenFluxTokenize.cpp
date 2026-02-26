#include "tokenize_common.hpp"
#include "tokenize_pipeline.hpp"

#include <iostream>

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    tokenflux::tokenize::Args args;
    if (!tokenflux::tokenize::parse_args(argc, argv, args))
    {
        return 0;
    }
    if (args.max_docs > 0)
    {
        std::cerr << "--max-docs is not supported in this C++ path yet. Please keep --max-docs 0.\n";
        return 1;
    }

    return tokenflux::tokenize::run_tokenize(args);
}
