#include "tokenize_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize
{

static void skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
}

static bool parse_hex4(const std::string &s, std::size_t i, std::uint32_t &out)
{
    if (i + 4 > s.size())
    {
        return false;
    }
    std::uint32_t val = 0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        char c = s[i + k];
        std::uint32_t v = 0;
        if (c >= '0' && c <= '9')
        {
            v = static_cast<std::uint32_t>(c - '0');
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = static_cast<std::uint32_t>(10 + c - 'a');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = static_cast<std::uint32_t>(10 + c - 'A');
        }
        else
        {
            return false;
        }
        val = (val << 4) | v;
    }
    out = val;
    return true;
}

static bool parse_json_string(const std::string &s, std::size_t &i, std::string &out)
{
    if (i >= s.size() || s[i] != '"')
    {
        return false;
    }
    ++i;
    out.clear();
    while (i < s.size())
    {
        char c = s[i++];
        if (c == '"')
        {
            return true;
        }
        if (c == '\\')
        {
            if (i >= s.size())
            {
                return false;
            }
            char esc = s[i++];
            switch (esc)
            {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                std::uint32_t cp = 0;
                if (!parse_hex4(s, i, cp))
                {
                    return false;
                }
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF)
                {
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u')
                    {
                        std::uint32_t low = 0;
                        if (parse_hex4(s, i + 2, low) && low >= 0xDC00 && low <= 0xDFFF)
                        {
                            i += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                    }
                }
                append_utf8(cp, out);
                break;
            }
            default:
                out.push_back(esc);
                break;
            }
        }
        else
        {
            out.push_back(c);
        }
    }
    return false;
}

static bool skip_json_value(const std::string &s, std::size_t &i);

static bool skip_json_number(const std::string &s, std::size_t &i)
{
    if (i < s.size() && (s[i] == '-' || s[i] == '+'))
    {
        ++i;
    }
    bool has_digit = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        has_digit = true;
        ++i;
    }
    if (i < s.size() && s[i] == '.')
    {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+'))
        {
            ++i;
        }
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    return has_digit;
}

static bool skip_json_literal(const std::string &s, std::size_t &i, const char *lit)
{
    std::size_t n = std::char_traits<char>::length(lit);
    if (i + n > s.size())
    {
        return false;
    }
    if (s.compare(i, n, lit) != 0)
    {
        return false;
    }
    i += n;
    return true;
}

static bool skip_json_object(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool skip_json_array(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool skip_json_value(const std::string &s, std::size_t &i)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    char c = s[i];
    if (c == '"')
    {
        std::string tmp;
        return parse_json_string(s, i, tmp);
    }
    if (c == '{')
    {
        return skip_json_object(s, i);
    }
    if (c == '[')
    {
        return skip_json_array(s, i);
    }
    if (c == 't')
    {
        return skip_json_literal(s, i, "true");
    }
    if (c == 'f')
    {
        return skip_json_literal(s, i, "false");
    }
    if (c == 'n')
    {
        return skip_json_literal(s, i, "null");
    }
    return skip_json_number(s, i);
}

static bool parse_json_uint(const std::string &s, std::size_t &i, std::uint32_t &out)
{
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
    {
        return false;
    }
    std::uint64_t val = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        val = val * 10 + static_cast<std::uint64_t>(s[i] - '0');
        if (val > std::numeric_limits<std::uint32_t>::max())
        {
            return false;
        }
        ++i;
    }
    out = static_cast<std::uint32_t>(val);
    return true;
}

static bool parse_json_double(const std::string &s, std::size_t &i, double &out)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    std::size_t start = i;
    if (s[i] == '+' || s[i] == '-')
    {
        ++i;
    }
    bool has_digit = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        has_digit = true;
        ++i;
    }
    if (i < s.size() && s[i] == '.')
    {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-'))
        {
            ++i;
        }
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (!has_digit)
    {
        return false;
    }
    try
    {
        out = std::stod(s.substr(start, i - start));
    }
    catch (...)
    {
        return false;
    }
    return true;
}

static bool parse_vocab_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        std::uint32_t val = 0;
        if (!parse_json_uint(s, i, val))
        {
            return false;
        }
        out.vocab[std::move(key)] = val;
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_merge_line(const std::string &line, std::pair<std::string, std::string> &out)
{
    auto pos = line.find(' ');
    if (pos == std::string::npos || pos + 1 >= line.size())
    {
        return false;
    }
    out.first = line.substr(0, pos);
    out.second = line.substr(pos + 1);
    return true;
}

static bool parse_merges_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] == '"')
        {
            std::string merge_line;
            if (!parse_json_string(s, i, merge_line))
            {
                return false;
            }
            std::pair<std::string, std::string> m;
            if (parse_merge_line(merge_line, m))
            {
                out.merges.push_back(std::move(m));
            }
        }
        else if (s[i] == '[')
        {
            ++i;
            skip_ws(s, i);
            std::string left;
            std::string right;
            bool ok = parse_json_string(s, i, left);
            skip_ws(s, i);
            if (ok && i < s.size() && s[i] == ',')
            {
                ++i;
                skip_ws(s, i);
                ok = parse_json_string(s, i, right);
            }
            while (i < s.size())
            {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    break;
                }
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    if (!skip_json_value(s, i))
                    {
                        return false;
                    }
                    continue;
                }
                return false;
            }
            if (ok && !left.empty() && !right.empty())
            {
                out.merges.push_back({std::move(left), std::move(right)});
            }
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static void set_model_type_from_string(const std::string &type, TokenizerData &out)
{
    if (type == "BPE")
    {
        out.model_type = ModelType::bpe;
        return;
    }
    if (type == "WordPiece")
    {
        out.model_type = ModelType::wordpiece;
        return;
    }
    if (type == "Unigram")
    {
        out.model_type = ModelType::unigram;
    }
}

static void set_pretokenizer_type_from_string(const std::string &type, TokenizerData &out)
{
    if (type == "ByteLevel")
    {
        out.pretokenizer_type = PretokenizerType::byte_level;
        return;
    }
    if (type == "WhitespaceSplit" || type == "Whitespace")
    {
        out.pretokenizer_type = PretokenizerType::whitespace;
    }
}

static bool parse_unigram_vocab_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    std::size_t index = 0;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] == '[')
        {
            ++i;
            skip_ws(s, i);
            std::string token;
            double score = 0.0;
            bool ok = parse_json_string(s, i, token);
            skip_ws(s, i);
            if (ok && i < s.size() && s[i] == ',')
            {
                ++i;
                skip_ws(s, i);
                ok = parse_json_double(s, i, score);
            }
            while (i < s.size())
            {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    break;
                }
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    if (!skip_json_value(s, i))
                    {
                        return false;
                    }
                    continue;
                }
                return false;
            }
            if (ok && !token.empty())
            {
                out.unigram_vocab.push_back({token, score});
                out.vocab[token] = static_cast<std::uint32_t>(index);
                ++index;
            }
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_pre_tokenizer_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    if (s.compare(i, 4, "null") == 0)
    {
        i += 4;
        return true;
    }
    if (s[i] != '{')
    {
        return skip_json_value(s, i);
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "type")
        {
            std::string type;
            if (!parse_json_string(s, i, type))
            {
                return false;
            }
            set_pretokenizer_type_from_string(type, out);
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_added_tokens_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] != '{')
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        else
        {
            ++i;
            std::string content;
            std::uint32_t id = 0;
            bool has_content = false;
            bool has_id = false;
            while (true)
            {
                skip_ws(s, i);
                if (i >= s.size())
                {
                    return false;
                }
                if (s[i] == '}')
                {
                    ++i;
                    break;
                }
                std::string key;
                if (!parse_json_string(s, i, key))
                {
                    return false;
                }
                skip_ws(s, i);
                if (i >= s.size() || s[i] != ':')
                {
                    return false;
                }
                ++i;
                if (key == "id")
                {
                    std::uint32_t x = 0;
                    if (!parse_json_uint(s, i, x))
                    {
                        return false;
                    }
                    id = x;
                    has_id = true;
                }
                else if (key == "content")
                {
                    std::string x;
                    if (!parse_json_string(s, i, x))
                    {
                        return false;
                    }
                    content = std::move(x);
                    has_content = true;
                }
                else if (!skip_json_value(s, i))
                {
                    return false;
                }
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == '}')
                {
                    ++i;
                    break;
                }
                return false;
            }
            if (has_content && has_id)
            {
                out.added_tokens[content] = id;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_model_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "vocab")
        {
            skip_ws(s, i);
            if (i < s.size() && s[i] == '{')
            {
                if (!parse_vocab_object(s, i, out))
                {
                    return false;
                }
            }
            else if (i < s.size() && s[i] == '[')
            {
                if (!parse_unigram_vocab_array(s, i, out))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        else if (key == "merges")
        {
            if (!parse_merges_array(s, i, out))
            {
                return false;
            }
        }
        else if (key == "unk_token")
        {
            std::string unk;
            if (!parse_json_string(s, i, unk))
            {
                return false;
            }
            out.unk_token = std::move(unk);
        }
        else if (key == "type")
        {
            std::string type;
            if (!parse_json_string(s, i, type))
            {
                return false;
            }
            set_model_type_from_string(type, out);
        }
        else if (key == "continuing_subword_prefix")
        {
            std::string prefix;
            if (!parse_json_string(s, i, prefix))
            {
                return false;
            }
            out.continuing_subword_prefix = std::move(prefix);
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_tokenizer_json(const std::string &content, TokenizerData &out)
{
    std::size_t i = 0;
    skip_ws(content, i);
    if (i >= content.size() || content[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(content, i);
        if (i >= content.size())
        {
            return false;
        }
        if (content[i] == '}')
        {
            ++i;
            break;
        }
        std::string key;
        if (!parse_json_string(content, i, key))
        {
            return false;
        }
        skip_ws(content, i);
        if (i >= content.size() || content[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "model")
        {
            if (!parse_model_object(content, i, out))
            {
                return false;
            }
        }
        else if (key == "added_tokens")
        {
            if (!parse_added_tokens_array(content, i, out))
            {
                return false;
            }
        }
        else if (key == "pre_tokenizer")
        {
            if (!parse_pre_tokenizer_object(content, i, out))
            {
                return false;
            }
        }
        else if (!skip_json_value(content, i))
        {
            return false;
        }
        skip_ws(content, i);
        if (i < content.size() && content[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < content.size() && content[i] == '}')
        {
            ++i;
            break;
        }
        return false;
    }
    for (const auto &kv : out.added_tokens)
    {
        out.vocab[kv.first] = kv.second;
    }
    return !out.vocab.empty();
}

bool TokenizerEncoder::load(const std::string &path, std::string &err)
{
    std::string content = read_file_all(path);
    if (content.empty())
    {
        err = "failed to read tokenizer file: " + path;
        return false;
    }
    TokenizerData data;
    data.vocab.reserve(65536);
    if (!parse_tokenizer_json(content, data))
    {
        err = "failed to parse tokenizer.json: " + path;
        return false;
    }
    vocab_ = std::move(data.vocab);
    model_type_ = data.model_type;
    pretokenizer_type_ = data.pretokenizer_type;
    continuing_subword_prefix_ = data.continuing_subword_prefix;
    if (continuing_subword_prefix_.empty())
    {
        continuing_subword_prefix_ = "##";
    }
    unk_token_ = data.unk_token;
    has_unk_ = false;
    if (!unk_token_.empty())
    {
        auto it = vocab_.find(unk_token_);
        if (it != vocab_.end())
        {
            has_unk_ = true;
            unk_id_ = it->second;
        }
    }

    auto cp_map = build_byte_to_unicode_cp();
    byte_to_unicode_ = build_byte_to_unicode_str(cp_map);

    symbols_.clear();
    symbol_to_id_.clear();
    merge_rules_.clear();
    unigram_tokens_.clear();
    unigram_index_.clear();

    if (model_type_ == ModelType::bpe)
    {
        symbols_.reserve(vocab_.size() + data.merges.size() + 256);
        symbol_to_id_.reserve(vocab_.size() + data.merges.size() + 256);
        merge_rules_.reserve(data.merges.size() * 13 / 10 + 8);

        if (pretokenizer_type_ == PretokenizerType::byte_level)
        {
            for (const auto &s : byte_to_unicode_)
            {
                ensure_symbol(s);
            }
        }
        for (const auto &kv : vocab_)
        {
            ensure_symbol(kv.first);
        }
        for (std::size_t rank = 0; rank < data.merges.size(); ++rank)
        {
            const auto &m = data.merges[rank];
            std::uint32_t left = ensure_symbol(m.first);
            std::uint32_t right = ensure_symbol(m.second);
            std::uint32_t merged = ensure_symbol(m.first + m.second);
            std::uint64_t key = pair_key(left, right);
            if (merge_rules_.find(key) == merge_rules_.end())
            {
                merge_rules_[key] = MergeRule{static_cast<std::uint32_t>(rank), merged};
            }
        }
    }
    else if (model_type_ == ModelType::unigram)
    {
        std::vector<UnigramEntry> entries = data.unigram_vocab;
        if (entries.empty())
        {
            std::vector<std::pair<std::string, std::uint32_t>> ordered;
            ordered.reserve(vocab_.size());
            for (const auto &kv : vocab_)
            {
                ordered.push_back(kv);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
            for (const auto &kv : ordered)
            {
                entries.push_back({kv.first, -1.0});
            }
        }
        unigram_tokens_.reserve(entries.size());
        for (const auto &entry : entries)
        {
            auto it = vocab_.find(entry.token);
            if (it == vocab_.end())
            {
                continue;
            }
            auto cps = split_codepoints_utf8(entry.token);
            if (cps.empty())
            {
                continue;
            }
            unigram_tokens_.push_back({entry.token, std::move(cps), entry.score, it->second});
        }
        for (std::size_t i = 0; i < unigram_tokens_.size(); ++i)
        {
            unigram_index_[unigram_tokens_[i].cps.front()].push_back(i);
        }
        for (auto &kv : unigram_index_)
        {
            auto &vec = kv.second;
            std::sort(vec.begin(), vec.end(), [&](std::size_t a, std::size_t b) {
                return unigram_tokens_[a].cps.size() > unigram_tokens_[b].cps.size();
            });
        }
    }

    return true;
}

std::size_t TokenizerEncoder::vocab_size() const
{
    return vocab_.size();
}

std::string TokenizerEncoder::model_name() const
{
    if (model_type_ == ModelType::bpe)
    {
        return "BPE";
    }
    if (model_type_ == ModelType::wordpiece)
    {
        return "WordPiece";
    }
    return "Unigram";
}

bool TokenizerEncoder::token_to_id(const std::string &token, std::uint32_t &id) const
{
    auto it = vocab_.find(token);
    if (it == vocab_.end())
    {
        return false;
    }
    id = it->second;
    return true;
}

void TokenizerEncoder::encode_text_append(const std::string &text,
                                          std::unordered_map<std::string, std::vector<std::uint32_t>> &cache,
                                          std::vector<std::uint32_t> &out_ids) const
{
    std::vector<std::string> pieces;
    if (pretokenizer_type_ == PretokenizerType::byte_level)
    {
        pieces = pretokenize(text);
    }
    else
    {
        pieces = split_whitespace_words(text);
    }
    for (const auto &piece : pieces)
    {
        if (piece.empty())
        {
            continue;
        }
        const std::vector<std::uint32_t> *ids = nullptr;
        if (model_type_ == ModelType::bpe)
        {
            std::string encoded = piece;
            if (pretokenizer_type_ == PretokenizerType::byte_level)
            {
                encoded = byte_level_encode(piece, byte_to_unicode_);
            }
            ids = &encode_piece_bpe(encoded, cache);
        }
        else if (model_type_ == ModelType::wordpiece)
        {
            ids = &encode_piece_wordpiece(piece, cache);
        }
        else
        {
            ids = &encode_piece_unigram(piece, cache);
        }
        if (ids)
        {
            out_ids.insert(out_ids.end(), ids->begin(), ids->end());
        }
    }
}

std::uint32_t TokenizerEncoder::ensure_symbol(const std::string &sym)
{
    auto it = symbol_to_id_.find(sym);
    if (it != symbol_to_id_.end())
    {
        return it->second;
    }
    std::uint32_t id = static_cast<std::uint32_t>(symbols_.size());
    symbols_.push_back(sym);
    symbol_to_id_.emplace(symbols_.back(), id);
    return id;
}

std::uint64_t TokenizerEncoder::pair_key(std::uint32_t a, std::uint32_t b)
{
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

const std::vector<std::uint32_t> &TokenizerEncoder::encode_piece_bpe(
    const std::string &encoded, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
{
    std::string cache_key = std::string("bpe:") + encoded;
    auto it_cache = cache.find(cache_key);
    if (it_cache != cache.end())
    {
        return it_cache->second;
    }

    std::vector<std::uint32_t> symbols;
    symbols.reserve(encoded.size());
    std::size_t i = 0;
    while (i < encoded.size())
    {
        std::size_t prev = i;
        std::uint32_t cp = 0;
        if (!next_codepoint(encoded, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        auto it_sym = symbol_to_id_.find(encoded.substr(prev, i - prev));
        if (it_sym == symbol_to_id_.end())
        {
            symbols.clear();
            break;
        }
        symbols.push_back(it_sym->second);
    }

    while (symbols.size() >= 2)
    {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best_pos = static_cast<std::size_t>(-1);
        std::uint32_t best_left = 0;
        std::uint32_t best_right = 0;
        std::uint32_t best_merged = 0;
        for (std::size_t pos = 0; pos + 1 < symbols.size(); ++pos)
        {
            std::uint64_t key = pair_key(symbols[pos], symbols[pos + 1]);
            auto it_rule = merge_rules_.find(key);
            if (it_rule == merge_rules_.end())
            {
                continue;
            }
            if (it_rule->second.rank < best_rank)
            {
                best_rank = it_rule->second.rank;
                best_pos = pos;
                best_left = symbols[pos];
                best_right = symbols[pos + 1];
                best_merged = it_rule->second.merged_symbol;
            }
        }
        if (best_pos == static_cast<std::size_t>(-1))
        {
            break;
        }

        std::vector<std::uint32_t> merged;
        merged.reserve(symbols.size());
        for (std::size_t p = 0; p < symbols.size();)
        {
            if (p + 1 < symbols.size() && symbols[p] == best_left && symbols[p + 1] == best_right)
            {
                merged.push_back(best_merged);
                p += 2;
            }
            else
            {
                merged.push_back(symbols[p]);
                ++p;
            }
        }
        symbols.swap(merged);
    }

    std::vector<std::uint32_t> ids;
    ids.reserve(symbols.size());
    for (std::uint32_t sid : symbols)
    {
        if (sid >= symbols_.size())
        {
            continue;
        }
        const auto &tok = symbols_[sid];
        auto it_vocab = vocab_.find(tok);
        if (it_vocab != vocab_.end())
        {
            ids.push_back(it_vocab->second);
        }
        else if (has_unk_)
        {
            ids.push_back(unk_id_);
        }
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(ids));
    return inserted.first->second;
}

const std::vector<std::uint32_t> &TokenizerEncoder::encode_piece_wordpiece(
    const std::string &piece, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
{
    std::string cache_key = std::string("wp:") + piece;
    auto it_cache = cache.find(cache_key);
    if (it_cache != cache.end())
    {
        return it_cache->second;
    }

    std::vector<std::uint32_t> ids;
    auto cps = split_codepoints_utf8(piece);
    if (!cps.empty())
    {
        std::size_t start = 0;
        bool fallback_to_unk = false;
        while (start < cps.size())
        {
            bool found = false;
            std::size_t best_end = start;
            std::uint32_t best_id = 0;
            for (std::size_t end = cps.size(); end > start; --end)
            {
                std::string cand;
                for (std::size_t k = start; k < end; ++k)
                {
                    cand += cps[k];
                }
                if (start > 0 && !continuing_subword_prefix_.empty())
                {
                    cand = continuing_subword_prefix_ + cand;
                }
                auto it = vocab_.find(cand);
                if (it != vocab_.end())
                {
                    best_id = it->second;
                    best_end = end;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                fallback_to_unk = true;
                break;
            }
            ids.push_back(best_id);
            start = best_end;
        }
        if (fallback_to_unk && has_unk_)
        {
            ids.clear();
            ids.push_back(unk_id_);
        }
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(ids));
    return inserted.first->second;
}

bool TokenizerEncoder::unigram_match(const std::vector<std::string> &token_cps, const std::vector<std::string> &word_cps,
                                     std::size_t pos)
{
    if (pos + token_cps.size() > word_cps.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < token_cps.size(); ++i)
    {
        if (token_cps[i] != word_cps[pos + i])
        {
            return false;
        }
    }
    return true;
}

const std::vector<std::uint32_t> &TokenizerEncoder::encode_piece_unigram(
    const std::string &piece, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
{
    std::string cache_key = std::string("uni:") + piece;
    auto it_cache = cache.find(cache_key);
    if (it_cache != cache.end())
    {
        return it_cache->second;
    }

    std::vector<std::uint32_t> ids;
    auto cps = split_codepoints_utf8(piece);
    if (!cps.empty() && !unigram_tokens_.empty())
    {
        const double neg_inf = -1e100;
        std::size_t n = cps.size();
        std::vector<double> best(n + 1, neg_inf);
        std::vector<int> prev_pos(n + 1, -1);
        std::vector<int> prev_tok(n + 1, -1);
        best[0] = 0.0;

        for (std::size_t i = 0; i < n; ++i)
        {
            if (best[i] <= neg_inf / 2.0)
            {
                continue;
            }
            auto it = unigram_index_.find(cps[i]);
            if (it == unigram_index_.end())
            {
                continue;
            }
            for (std::size_t tok_idx : it->second)
            {
                const auto &tok = unigram_tokens_[tok_idx];
                if (!unigram_match(tok.cps, cps, i))
                {
                    continue;
                }
                std::size_t j = i + tok.cps.size();
                double cand = best[i] + tok.score;
                if (cand > best[j])
                {
                    best[j] = cand;
                    prev_pos[j] = static_cast<int>(i);
                    prev_tok[j] = static_cast<int>(tok_idx);
                }
            }
        }

        if (best[n] <= neg_inf / 2.0)
        {
            if (has_unk_)
            {
                ids.push_back(unk_id_);
            }
        }
        else
        {
            std::vector<std::uint32_t> rev;
            int cur = static_cast<int>(n);
            while (cur > 0)
            {
                int t = prev_tok[static_cast<std::size_t>(cur)];
                int p = prev_pos[static_cast<std::size_t>(cur)];
                if (t < 0 || p < 0)
                {
                    break;
                }
                rev.push_back(unigram_tokens_[static_cast<std::size_t>(t)].id);
                cur = p;
            }
            ids.assign(rev.rbegin(), rev.rend());
        }
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(ids));
    return inserted.first->second;
}

} // namespace tokenflux::tokenize
