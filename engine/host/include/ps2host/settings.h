/*
 * The port's own settings: the PC enhancements a player chooses (fullscreen,
 * ...), kept between runs in a small text file of `key=value` lines and
 * written whenever one changes. A game's hooks offer them in its own menus
 * (Burnout 3: port/overrides.cpp, "PC SETTINGS"); the host applies them.
 */

#ifndef PS2HOST_SETTINGS_H
#define PS2HOST_SETTINGS_H

#include <string>
#include <vector>

namespace ps2host {

/* Reads the settings from `path` (none yet is no error) and keeps it as the
   file later changes go to. */
void LoadSettings(const std::string& path);
/* Where they are kept ("" before LoadSettings). */
const std::string& SettingsPath();

/* A setting's value, or `fallback` when it has none. */
std::string Setting(const std::string& key, const std::string& fallback = "");
bool SettingOn(const std::string& key, bool fallback = false);
/* Sets one and writes the file. */
void SetSetting(const std::string& key, const std::string& value);
void SetSettingOn(const std::string& key, bool on);
/* The settings the file gives that differ from their defaults (each as
   Setting was first asked for it; "not read yet" for one no code has asked
   for so far), or that none were read: a line for the console. */
std::string SettingsReport();

/* The run options the settings add when the port is played (started with no
   arguments, main.cpp). vu_record=on: the vector units' microprograms the
   game runs are recorded into vu_programs.txt beside the settings
   (--vu-record, added to at each run), for the port's next build to compile
   (port.py, vurecomp.py) - interpreted meanwhile (--vu-interpret), so that
   every jump is seen; slower while it is on. A program never recorded runs
   in the interpreter in any build. */
std::vector<std::string> PlaySettingArguments();

}  // namespace ps2host

#endif  /* PS2HOST_SETTINGS_H */
