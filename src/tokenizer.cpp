#include "tokenflux/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace tokenflux {

namespace {
std::vector<std::string> split_ws(std::string_view text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}
} // namespace

Tokenizer::Tokenizer() {
    vocab_.push_back(cfg_.unk_token);
    rebuild_index();
}

bool Tokenizer::load_hf_tokenizer_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    nlohmann::json j;
    in >> j;

    vocab_.clear();
    auto vocab_obj = j["model"]["vocab"];
    std::vector<std::pair<int, std::string>> id_token;
    id_token.reserve(vocab_obj.size());
    for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it) {
        id_token.emplace_back(it.value().get<int>(), it.key());
    }
    std::sort(id_token.begin(), id_token.end());
    for (auto& [id, tok] : id_token) {
        if (static_cast<size_t>(id) >= vocab_.size()) vocab_.resize(id + 1, cfg_.unk_token);
        vocab_[id] = tok;
    }

    merges_.clear();
    merge_rank_.clear();
    int rank = 0;
    if (j["model"].contains("merges")) {
        for (const auto& m : j["model"]["merges"]) {
            std::string line = m.get<std::string>();
            auto pos = line.find(' ');
            if (pos == std::string::npos) continue;
            auto a = line.substr(0, pos);
            auto b = line.substr(pos + 1);
            merges_.push_back({a, b});
            merge_rank_[a + "\t" + b] = rank++;
        }
    }

    rebuild_index();
    return true;
}

bool Tokenizer::save_hf_tokenizer_json(const std::string& path) const {
    nlohmann::json j;
    j["model"]["type"] = "BPE";
    for (size_t i = 0; i < vocab_.size(); ++i) {
        j["model"]["vocab"][vocab_[i]] = static_cast<int>(i);
    }
    for (const auto& [a, b] : merges_) {
        j["model"]["merges"].push_back(a + " " + b);
    }
    std::ofstream out(path);
    if (!out) return false;
    out << std::setw(2) << j;
    return true;
}

bool Tokenizer::load_bpe_files(const std::string& vocab_json, const std::string& merges_txt) {
    std::ifstream vin(vocab_json);
    if (!vin) return false;
    nlohmann::json j;
    vin >> j;

    std::vector<std::pair<int, std::string>> id_token;
    for (auto it = j.begin(); it != j.end(); ++it) {
        id_token.emplace_back(it.value().get<int>(), it.key());
    }
    std::sort(id_token.begin(), id_token.end());
    vocab_.assign(id_token.size(), cfg_.unk_token);
    for (auto& [id, tok] : id_token) {
        if (id >= 0 && static_cast<size_t>(id) < vocab_.size()) vocab_[id] = tok;
    }

    std::ifstream min(merges_txt);
    if (!min) return false;
    merges_.clear();
    merge_rank_.clear();
    std::string line;
    int rank = 0;
    while (std::getline(min, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        merges_.push_back({a, b});
        merge_rank_[a + "\t" + b] = rank++;
    }
    rebuild_index();
    return true;
}

bool Tokenizer::save_bpe_files(const std::string& vocab_json, const std::string& merges_txt) const {
    nlohmann::json j;
    for (size_t i = 0; i < vocab_.size(); ++i) j[vocab_[i]] = static_cast<int>(i);
    {
        std::ofstream out(vocab_json);
        if (!out) return false;
        out << std::setw(2) << j;
    }
    {
        std::ofstream out(merges_txt);
        if (!out) return false;
        out << "#version: 0.2\n";
        for (const auto& [a, b] : merges_) out << a << " " << b << "\n";
    }
    return true;
}

std::vector<std::string> Tokenizer::byte_encode_word(std::string_view word) const {
    std::vector<std::string> pieces;
    pieces.reserve(word.size());
    for (unsigned char ch : word) {
        std::ostringstream oss;
        oss << "<0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(ch) << ">";
        pieces.push_back(oss.str());
    }
    return pieces;
}

std::vector<std::string> Tokenizer::apply_bpe(std::vector<std::string> pieces) const {
    if (pieces.size() < 2 || merge_rank_.empty()) return pieces;
    while (pieces.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = pieces.size();
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            auto it = merge_rank_.find(pieces[i] + "\t" + pieces[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == pieces.size()) break;
        pieces[best_idx] += pieces[best_idx + 1];
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
    }
    return pieces;
}

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    std::vector<int32_t> ids;
    auto words = split_ws(text);
    for (const auto& w : words) {
        auto pieces = apply_bpe(byte_encode_word(w));
        for (const auto& p : pieces) ids.push_back(token_to_id(p));
    }
    return ids;
}

std::string Tokenizer::detokenize(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += id_to_token(id);
    return out;
}

int32_t Tokenizer::token_to_id(std::string_view token) const {
    auto it = token_to_id_.find(std::string(token));
    return it == token_to_id_.end() ? cfg_.unk_id : it->second;
}

std::string Tokenizer::id_to_token(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) return cfg_.unk_token;
    return vocab_[id];
}

void Tokenizer::set_vocab(std::vector<std::string> vocab) {
    vocab_ = std::move(vocab);
    rebuild_index();
}

void Tokenizer::set_merges(std::vector<std::pair<std::string, std::string>> merges) {
    merges_ = std::move(merges);
    merge_rank_.clear();
    for (size_t i = 0; i < merges_.size(); ++i) {
        merge_rank_[merges_[i].first + "\t" + merges_[i].second] = static_cast<int32_t>(i);
    }
}

void Tokenizer::rebuild_index() {
    token_to_id_.clear();
    for (size_t i = 0; i < vocab_.size(); ++i) token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
}

} // namespace tokenflux
