#include "input_source.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "tokenflux_lib.hpp"

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#endif

namespace
{
std::string trim_ascii(const std::string &value)
{
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
    {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        --end;
    }
    return value.substr(start, end - start);
}

bool starts_with_case_insensitive(const std::string &value, const std::string &prefix)
{
    if (value.size() < prefix.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < prefix.size(); ++i)
    {
        char a = static_cast<char>(std::tolower(static_cast<unsigned char>(value[i])));
        char b = static_cast<char>(std::tolower(static_cast<unsigned char>(prefix[i])));
        if (a != b)
        {
            return false;
        }
    }
    return true;
}

bool is_absolute_windows_path(const std::string &value)
{
    return value.size() >= 3 && std::isalpha(static_cast<unsigned char>(value[0])) && value[1] == ':' &&
           (value[2] == '\\' || value[2] == '/');
}

bool is_probably_url(const std::string &value)
{
    return starts_with_case_insensitive(value, "http://") || starts_with_case_insensitive(value, "https://") ||
           starts_with_case_insensitive(value, "file://");
}

std::string percent_decode(const std::string &value)
{
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == '%' && i + 2 < value.size())
        {
            auto from_hex = [](char c) -> int {
                if (c >= '0' && c <= '9')
                {
                    return c - '0';
                }
                if (c >= 'a' && c <= 'f')
                {
                    return 10 + c - 'a';
                }
                if (c >= 'A' && c <= 'F')
                {
                    return 10 + c - 'A';
                }
                return -1;
            };
            int hi = from_hex(value[i + 1]);
            int lo = from_hex(value[i + 2]);
            if (hi >= 0 && lo >= 0)
            {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        if (value[i] == '+')
        {
            out.push_back(' ');
            continue;
        }
        out.push_back(value[i]);
    }
    return out;
}

std::string decode_file_url(const std::string &value)
{
    std::string rest = value.substr(7);
    if (starts_with_case_insensitive(rest, "localhost/"))
    {
        rest = rest.substr(10);
    }
    if (rest.size() >= 3 && rest[0] == '/' && std::isalpha(static_cast<unsigned char>(rest[1])) && rest[2] == ':')
    {
        rest.erase(0, 1);
    }
    return percent_decode(rest);
}

std::string lower_ascii(std::string value)
{
    for (char &ch : value)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string strip_query_fragment(const std::string &value)
{
    std::size_t pos = value.find_first_of("?#");
    if (pos == std::string::npos)
    {
        return value;
    }
    return value.substr(0, pos);
}

std::string remote_extension_from_url(const std::string &value)
{
    std::string clean = strip_query_fragment(value);
    std::size_t slash = clean.find_last_of('/');
    std::string leaf = slash == std::string::npos ? clean : clean.substr(slash + 1);
    if (leaf.empty())
    {
        return ".bin";
    }
    static const std::vector<std::string> known_exts = {".jsonl.gz", ".json.gz", ".jsonl.xz", ".json.xz", ".jsonl",
                                                        ".json",    ".ndjson",  ".parquet",  ".gz",       ".xz"};
    std::string lower_leaf = lower_ascii(leaf);
    for (const auto &ext : known_exts)
    {
        if (lower_leaf.size() >= ext.size() &&
            lower_leaf.compare(lower_leaf.size() - ext.size(), ext.size(), ext) == 0)
        {
            return ext;
        }
    }
    std::size_t dot = leaf.find_last_of('.');
    if (dot == std::string::npos)
    {
        return ".bin";
    }
    return leaf.substr(dot);
}

std::string make_remote_cache_name(const std::string &url)
{
    std::size_t h = std::hash<std::string>{}(normalize_input_id(url));
    std::ostringstream oss;
    oss << "remote_" << std::hex << h << remote_extension_from_url(url);
    return oss.str();
}

bool stat_file_path(const std::string &path, std::uint64_t &file_size, std::int64_t &mtime)
{
    std::error_code ec;
    file_size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        return false;
    }
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec)
    {
        return false;
    }
    mtime = std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    return true;
}

std::vector<std::string> parse_list_entries(const std::string &payload)
{
    std::vector<std::string> out;
    std::istringstream iss(payload);
    std::string line;
    while (std::getline(iss, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        std::string trimmed = trim_ascii(line);
        if (trimmed.empty() || trimmed[0] == '#')
        {
            continue;
        }
        out.push_back(trimmed);
    }
    return out;
}

std::filesystem::path parent_dir_of_local_list(const std::string &path)
{
    std::error_code ec;
    auto abs = std::filesystem::absolute(path, ec);
    if (ec)
    {
        return std::filesystem::path(path).parent_path();
    }
    return abs.parent_path();
}

std::string parent_url_of_remote_list(const std::string &url)
{
    std::size_t slash = url.find_last_of('/');
    if (slash == std::string::npos)
    {
        return url;
    }
    return url.substr(0, slash + 1);
}

std::string resolve_list_entry(const std::string &list_ref, bool list_is_remote, const std::string &entry)
{
    if (entry.empty() || is_probably_url(entry) || is_absolute_windows_path(entry) ||
        (!entry.empty() && (entry[0] == '/' || entry[0] == '\\')))
    {
        return entry;
    }

    if (list_is_remote)
    {
        return parent_url_of_remote_list(list_ref) + entry;
    }

    std::filesystem::path base_dir = parent_dir_of_local_list(list_ref);
    return (base_dir / entry).string();
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string &value)
{
    if (value.empty())
    {
        return {};
    }
    int needed = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0);
    if (needed <= 0)
    {
        return {};
    }
    std::wstring out(static_cast<std::size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), out.data(), needed);
    return out;
}

bool download_http_to_bytes(const std::string &url, std::vector<std::uint8_t> &bytes, std::string &err)
{
    bytes.clear();

    std::wstring wide_url = utf8_to_wide(url);
    if (wide_url.empty())
    {
        err = "failed to convert URL to wide string: " + url;
        return false;
    }

    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwSchemeLength = static_cast<DWORD>(-1);
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    parts.dwExtraInfoLength = static_cast<DWORD>(-1);
    if (!WinHttpCrackUrl(wide_url.c_str(), static_cast<DWORD>(wide_url.size()), 0, &parts))
    {
        err = "failed to parse URL: " + url;
        return false;
    }

    std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    if (parts.dwExtraInfoLength > 0 && parts.lpszExtraInfo)
    {
        path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);
    }
    if (path.empty())
    {
        path = L"/";
    }

    HINTERNET session = WinHttpOpen(L"TokenFlux/0.3.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session)
    {
        err = "WinHttpOpen failed";
        return false;
    }

    HINTERNET connect = WinHttpConnect(session, host.c_str(), parts.nPort, 0);
    if (!connect)
    {
        WinHttpCloseHandle(session);
        err = "WinHttpConnect failed";
        return false;
    }

    DWORD flags = parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET request = WinHttpOpenRequest(connect, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                           WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!request)
    {
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        err = "WinHttpOpenRequest failed";
        return false;
    }

    bool ok = false;
    if (WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
        WinHttpReceiveResponse(request, nullptr))
    {
        DWORD status = 0;
        DWORD status_size = sizeof(status);
        if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                                 &status, &status_size, WINHTTP_NO_HEADER_INDEX))
        {
            err = "failed to query HTTP status: " + url;
        }
        else if (status < 200 || status >= 300)
        {
            err = "HTTP GET failed with status " + std::to_string(status) + ": " + url;
        }
        else
        {
            ok = true;
            while (true)
            {
                DWORD available = 0;
                if (!WinHttpQueryDataAvailable(request, &available))
                {
                    ok = false;
                    err = "WinHttpQueryDataAvailable failed: " + url;
                    break;
                }
                if (available == 0)
                {
                    break;
                }
                std::size_t offset = bytes.size();
                bytes.resize(offset + static_cast<std::size_t>(available));
                DWORD read = 0;
                if (!WinHttpReadData(request, bytes.data() + offset, available, &read))
                {
                    ok = false;
                    err = "WinHttpReadData failed: " + url;
                    break;
                }
                bytes.resize(offset + static_cast<std::size_t>(read));
            }
        }
    }
    else
    {
        err = "HTTP request failed: " + url;
    }

    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    return ok;
}
#endif

bool read_uri_text(const std::string &uri, std::string &payload, std::string &err)
{
    if (is_remote_http_url(uri))
    {
#ifdef _WIN32
        std::vector<std::uint8_t> bytes;
        if (!download_http_to_bytes(uri, bytes, err))
        {
            return false;
        }
        payload.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        return true;
#else
        err = "remote HTTP input is only implemented on Windows in this build";
        return false;
#endif
    }

    std::string path = is_file_url(uri) ? decode_file_url(uri) : uri;
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        err = "failed to open list file: " + uri;
        return false;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    payload = oss.str();
    return static_cast<bool>(in) || in.eof();
}

bool materialize_remote_input(const std::string &url, const std::string &remote_cache_dir, InputSource &source, std::string &err)
{
#ifdef _WIN32
    std::error_code ec;
    std::filesystem::create_directories(remote_cache_dir, ec);
    if (ec)
    {
        err = "failed to create remote cache dir: " + remote_cache_dir;
        return false;
    }

    std::filesystem::path local_path = std::filesystem::path(remote_cache_dir) / make_remote_cache_name(url);
    if (!std::filesystem::exists(local_path))
    {
        std::vector<std::uint8_t> bytes;
        if (!download_http_to_bytes(url, bytes, err))
        {
            return false;
        }
        std::ofstream out(local_path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            err = "failed to create remote cache file: " + local_path.string();
            return false;
        }
        if (!bytes.empty())
        {
            out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        if (!out)
        {
            err = "failed to write remote cache file: " + local_path.string();
            return false;
        }
    }

    source.source = url;
    source.local_path = local_path.string();
    source.normalized_id = normalize_input_id(url);
    source.remote = true;
    if (!stat_file_path(source.local_path, source.file_size, source.file_mtime))
    {
        err = "failed to stat remote cache file: " + source.local_path;
        return false;
    }
    return true;
#else
    (void)url;
    (void)remote_cache_dir;
    (void)source;
    err = "remote HTTP input is only implemented on Windows in this build";
    return false;
#endif
}

bool materialize_local_input(const std::string &raw_value, InputSource &source, std::string &err)
{
    std::string path = is_file_url(raw_value) ? decode_file_url(raw_value) : raw_value;
    if (!std::filesystem::exists(path))
    {
        err = "input does not exist: " + raw_value;
        return false;
    }
    source.source = raw_value;
    source.local_path = path;
    source.normalized_id = normalize_input_id(raw_value);
    source.remote = false;
    if (!stat_file_path(path, source.file_size, source.file_mtime))
    {
        err = "failed to stat input file: " + path;
        return false;
    }
    return true;
}
} // namespace

bool is_remote_http_url(const std::string &value)
{
    return starts_with_case_insensitive(value, "http://") || starts_with_case_insensitive(value, "https://");
}

bool is_file_url(const std::string &value)
{
    return starts_with_case_insensitive(value, "file://");
}

std::string normalize_input_id(const std::string &value)
{
    std::string out = value;
    if (is_file_url(out))
    {
        out = decode_file_url(out);
    }
    std::replace(out.begin(), out.end(), '\\', '/');
#ifdef _WIN32
    out = lower_ascii(out);
#endif
    return out;
}

bool resolve_input_sources(const std::string &data_glob, const std::string &data_list,
                           const std::string &remote_cache_dir, std::vector<InputSource> &sources, std::string &err)
{
    sources.clear();

    std::vector<std::string> entries;
    if (!data_list.empty())
    {
        std::string payload;
        if (!read_uri_text(data_list, payload, err))
        {
            return false;
        }
        auto listed = parse_list_entries(payload);
        bool list_is_remote = is_remote_http_url(data_list);
        for (const auto &entry : listed)
        {
            std::string resolved = resolve_list_entry(data_list, list_is_remote, entry);
            if (!is_probably_url(resolved))
            {
                auto expanded = expand_data_glob(resolved);
                if (!expanded.empty())
                {
                    entries.insert(entries.end(), expanded.begin(), expanded.end());
                    continue;
                }
            }
            entries.push_back(resolved);
        }
    }
    else if (!data_glob.empty())
    {
        entries = expand_data_glob(data_glob);
    }

    if (entries.empty())
    {
        if (!data_list.empty())
        {
            err = "no inputs found from data list: " + data_list;
        }
        return !data_list.empty() ? false : true;
    }

    std::vector<std::string> seen;
    seen.reserve(entries.size());
    for (const auto &entry : entries)
    {
        InputSource source;
        std::string local_err;
        bool ok = false;
        if (is_remote_http_url(entry))
        {
            ok = materialize_remote_input(entry, remote_cache_dir, source, local_err);
        }
        else
        {
            ok = materialize_local_input(entry, source, local_err);
        }
        if (!ok)
        {
            err = local_err;
            return false;
        }
        if (std::find(seen.begin(), seen.end(), source.normalized_id) != seen.end())
        {
            continue;
        }
        seen.push_back(source.normalized_id);
        sources.push_back(std::move(source));
    }

    return true;
}
