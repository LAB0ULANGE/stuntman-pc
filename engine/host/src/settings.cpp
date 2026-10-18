/*
 * The port's settings (settings.h): a map of keys to values, read from and
 * written to one text file. Lines are `key=value`; anything else (comments,
 * blank lines) is left out when the file is written again.
 */

#include "ps2host/settings.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

namespace ps2host {

namespace {

std::mutex g_lock;
std::map<std::string, std::string> g_values;
std::string g_path;
// What each setting is when the file does not say (Setting's fallback, the
// first one asked for): ReportSettings's "differs from".
std::map<std::string, std::string> g_defaults;
bool g_loaded = false;

std::string Trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

void Write() {
    if (g_path.empty()) return;
    std::ofstream out(g_path, std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "[settings] cannot write %s\n", g_path.c_str());
        return;
    }
    out << "# The PC port's settings (PC SETTINGS in the main menu).\n";
    for (const auto& [key, value] : g_values) out << key << '=' << value << '\n';
}

}  // namespace

void LoadSettings(const std::string& path) {
    std::lock_guard<std::mutex> hold(g_lock);
    g_path = path;
    g_values.clear();
    g_loaded = true;
    std::ifstream in(path);
    // Which file a run takes its settings from, said at once: one beside
    // --memcard's directory, not the executable's, is easily missed.
    if (in) std::fprintf(stderr, "[settings] read from %s\n", path.c_str());
    else std::fprintf(stderr, "[settings] no file (%s): the defaults\n", path.c_str());
    std::string line;
    while (std::getline(in, line)) {
        const auto equals = line.find('=');
        if (line.empty() || line[0] == '#' || equals == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, equals));
        if (!key.empty()) g_values[key] = Trim(line.substr(equals + 1));
    }
}

const std::string& SettingsPath() { return g_path; }

std::string Setting(const std::string& key, const std::string& fallback) {
    std::lock_guard<std::mutex> hold(g_lock);
    g_defaults.emplace(key, fallback);
    const auto found = g_values.find(key);
    return found == g_values.end() ? fallback : found->second;
}

bool SettingOn(const std::string& key, bool fallback) {
    const std::string value = Setting(key, fallback ? "on" : "off");
    return value == "on" || value == "1" || value == "true" || value == "yes";
}

void SetSetting(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> hold(g_lock);
    if (g_values.count(key) && g_values[key] == value) return;
    g_values[key] = value;
    Write();
}

void SetSettingOn(const std::string& key, bool on) { SetSetting(key, on ? "on" : "off"); }

std::string SettingsReport() {
    std::lock_guard<std::mutex> hold(g_lock);
    if (!g_loaded) return "[settings] none read (no file looked for): the defaults";
    std::string out;
    for (const auto& [key, value] : g_values) {
        const auto asked = g_defaults.find(key);
        if (asked != g_defaults.end() && asked->second == value) continue;
        out += out.empty() ? " " : ", ";
        out += key + "=" + value;
        out += asked == g_defaults.end() ? " (not read yet)"
               : asked->second.empty() ? " (default: none)" : " (default " + asked->second + ")";
    }
    return "[settings] differing from the defaults:" + (out.empty() ? std::string(" none") : out);
}

std::vector<std::string> PlaySettingArguments() {
    std::vector<std::string> arguments;
    if (SettingOn("vu_record")) {
        const std::filesystem::path beside = std::filesystem::path(SettingsPath()).parent_path();
        arguments = {"--vu-interpret", "--vu-record", (beside / "vu_programs.txt").string()};
    }
    return arguments;
}

}  // namespace ps2host
