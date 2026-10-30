/*
 * Every game port's entry point. The game supplies its translated functions (the
 * table engine/tools/port.py generates) and its overrides; the rest is shared.
 */

#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <io.h>
#include <sys/stat.h>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

extern const ps2host::FunctionEntry kGuestFunctions[];
extern const std::size_t kGuestFunctionCount;

/* The game's host implementations of guest functions (its port's overrides.cpp). */
void RegisterGameOverrides();

namespace {

/* The instruction set the port is built for (PS2_ARCH, engine/host/
   CMakeLists.txt: AVX2, AVX, or SSE2 - with SSE4.1, which the VU's lanes
   take without /arch, ps2rt/vu_math.h): a processor without it is told so
   rather than stopped by an illegal instruction somewhere. */
#if defined(__AVX2__)
const char* const kNeeds = "AVX2 (Intel Haswell, AMD Excavator or later)";
#elif defined(__AVX__)
const char* const kNeeds = "AVX (Intel Sandy Bridge, AMD Bulldozer or later)";
#else
const char* const kNeeds = "SSE4.1 (Intel Core 2 of 2008, AMD Bulldozer or later)";
#endif

bool HasInstructionSet() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4];
    __cpuid(info, 0);
    const int top = info[0];
    __cpuid(info, 1);
    if (!((info[2] >> 19) & 1)) return false;       // SSE4.1
#if defined(__AVX__)
    const bool osxsave = (info[2] >> 27) & 1, avx = (info[2] >> 28) & 1;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 6) != 6) return false;        // the OS saves the YMM registers
#endif
#if defined(__AVX2__)
    if (top < 7) return false;
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
#else
    (void)top;
    return true;
#endif
#else
    return true;
#endif
}

std::string g_log;     // the log a run started without arguments writes

#ifdef PS2HOST_GAME_EXECUTABLE
/* Whether `path` is an image of the game's disc: the executable it boots is
   the one the port was translated from (game.json's). */
bool IsGameDisc(const std::string& path) {
    ps2host::Disc disc;
    if (!disc.Open(path)) return false;
    std::string boot = disc.BootPath();
    const std::size_t semicolon = boot.find(';');
    if (semicolon != std::string::npos) boot.resize(semicolon);
    while (!boot.empty() && (boot[0] == '\\' || boot[0] == '/')) boot.erase(0, 1);
    return _stricmp(boot.c_str(), PS2HOST_GAME_EXECUTABLE) == 0;
}

/* The player's disc image, from a file dialog; "" if they cancel. */
std::string PickDisc() {
    char file[MAX_PATH] = {};
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof dialog;
    dialog.lpstrFilter = "Disc images (*.iso; *.cue; *.bin)\0*.iso;*.cue;*.bin\0All files (*.*)\0*.*\0";
    dialog.lpstrFile = file;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrTitle = "Choose your " PS2HOST_GAME_NAME " disc image";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&dialog) ? std::string(file) : std::string();
}
#endif

/* Started without arguments - from Explorer, a shortcut - the port plays its
   game from the player's own disc image: the one chosen before (kept in the
   PC settings, beside the executable), or asked for and checked to be the
   game's disc; the memory card a directory beside it too. In a window, with
   sound, on the GPU, with no time limit (the default is for test runs).
   An empty list: the player gave none. */
std::vector<std::string> PlayArguments(const char* self) {
#ifdef PS2HOST_GAME_EXECUTABLE
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::filesystem::path here = std::filesystem::path(exe).parent_path();
    ps2host::LoadSettings((here / "pc_settings.ini").string());     // where --memcard puts them
    std::string disc = ps2host::Setting("disc");
    while (!IsGameDisc(disc)) {
        if (!disc.empty() && GetFileAttributesA(disc.c_str()) != INVALID_FILE_ATTRIBUTES)
            MessageBoxA(nullptr, ("This is not an image of the " PS2HOST_GAME_NAME " disc "
                                  "(" PS2HOST_GAME_EXECUTABLE "):\n\n" + disc).c_str(),
                        PS2HOST_GAME_NAME, MB_OK | MB_ICONWARNING);
        disc = PickDisc();
        if (disc.empty()) return {};
    }
    if (disc != ps2host::Setting("disc")) ps2host::SetSetting("disc", disc);
    // No terminal to read: what the run says goes to a log beside it. Opened
    // once, shared (fopen's would lock the file to one stream), and given to
    // both: their lines in the order they were written.
    g_log = std::filesystem::path(exe).stem().string() + ".log";
    const std::string log = (here / g_log).string();
    std::FILE* stream = nullptr;
    freopen_s(&stream, "NUL", "w", stdout);    // streams with descriptors, in a windowed program
    freopen_s(&stream, "NUL", "w", stderr);
    const int file = _open(log.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_TEXT, _S_IREAD | _S_IWRITE);
    if (file >= 0) {
        _dup2(file, _fileno(stdout));
        _dup2(file, _fileno(stderr));
        _close(file);
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a run ended by force loses nothing
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::fprintf(stderr, "[port] disc %s\n", disc.c_str());
    std::vector<std::string> arguments = {self, "--disc", disc, "--memcard", (here / "memcard").string(),
                                          "--window", "--gpu", "--audio", "--timeout", "315360000"};  // ten years
    for (const std::string& extra : ps2host::PlaySettingArguments()) arguments.push_back(extra);
    for (std::size_t i = 1; i < arguments.size(); ++i) std::fprintf(stderr, i == 1 ? "[port] %s" : " %s", arguments[i].c_str());
    std::fprintf(stderr, "\n");
    return arguments;
#else
    return {self};
#endif
}

}  // namespace

/* A port built as a windowed program has no console of its own. Started from
   a terminal its output goes to that one; redirected (a file, a pipe: the
   tools' runs), where it was sent. */
void AttachTerminal() {
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != nullptr && out != INVALID_HANDLE_VALUE) return;
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    std::FILE* stream = nullptr;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
}

int main(int argc, char** argv) {
    AttachTerminal();
    std::vector<std::string> play;
    std::vector<char*> play_argv;
    if (argc == 1) {
        play = PlayArguments(argv[0]);
        if (play.empty()) return 0;
        for (std::string& arg : play) play_argv.push_back(arg.data());
        play_argv.push_back(nullptr);
        argc = int(play.size());
        argv = play_argv.data();
    }
    if (!HasInstructionSet()) {
        std::fprintf(stderr, "This build needs a processor with %s.\n", kNeeds);
        return 70;
    }
#ifdef PS2HOST_GAME_TITLE
    ps2host::SetGameTitle(PS2HOST_GAME_TITLE);      // the window's (game.json's "title")
#endif
#ifdef PS2HOST_GAME_VERSION
    std::fprintf(stderr, "[port] %s, version %s\n", PS2HOST_GAME_TITLE, PS2HOST_GAME_VERSION);
#endif
    ps2host::RegisterFunctions(kGuestFunctions, kGuestFunctionCount);
    RegisterGameOverrides();
    const int code = ps2host::Main(argc, argv);
    if (!play.empty() && code != 0) {         // closing the window ends a run with 0
        std::fflush(nullptr);
        const std::string text = "The game stopped because of an error. What happened is in " +
                                 g_log + ", beside the game.";
        MessageBoxA(nullptr, text.c_str(), ps2host::GameTitle().empty() ? "PS2 port" : ps2host::GameTitle().c_str(),
                    MB_OK | MB_ICONERROR);
    }
    // What must be finished is (the report written, the settings and the
    // memory card saved as they changed, the WAV's header here); the rest -
    // the window, the GPU device, the threads - goes with the process.
    // Taken down one by one at exit, the Vulkan device's could hang
    // (vkDestroyDevice with the driver's threads and the renderer's images
    // still about), leaving a window that never closed.
    ps2host::CloseAudio();
    std::fflush(nullptr);
    TerminateProcess(GetCurrentProcess(), UINT(code));
    return code;
}
