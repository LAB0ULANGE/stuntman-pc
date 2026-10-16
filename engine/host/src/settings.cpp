/*
 * The port's settings (settings.h): a map of keys to values, read from and
 * written to one text file. Lines are `key=value`; anything else (comments,
 * blank lines) is left out when the file is written again.
 */

#include "ps2host/settings.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>

namespace ps2host {

namespace {

std::mutex g_lock;
std::map<std::string, std::string> g_values;
std::string g_path;

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
    std::ifstream in(path);
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

}  // namespace ps2host
