/*
 * The port's window (--window): the GS's displayed picture, every vertical
 * blank, scaled to the window, and the keyboard the controller reads.
 *
 * With the GPU display built in (PS2HOST_VULKAN, engine/gpu) the window is an
 * SDL3 window presenting through Vulkan; without it, or when no Vulkan device
 * can be had, a plain Win32 window drawn with GDI. Both are pumped on the host's
 * thread at each frame (FrameDone).
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <functional>
#include <vector>

#include "ps2host/gs.h"
#include "ps2host/timeline.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/texture_pack.h"

#ifdef PS2HOST_VULKAN
#include <SDL3/SDL_scancode.h>

#include "ps2gpu/display.h"
#endif

namespace ps2host {

namespace {

HWND g_window = nullptr;
bool g_closed = false;

#ifdef PS2HOST_VULKAN
std::unique_ptr<ps2gpu::Display> g_display;

// The keys the controller reads, as Win32 virtual keys, by position.
int Scancode(int key) {
    switch (key) {
        case VK_BACK: return SDL_SCANCODE_BACKSPACE;
        case VK_RETURN: return SDL_SCANCODE_RETURN;
        case VK_UP: return SDL_SCANCODE_UP;
        case VK_DOWN: return SDL_SCANCODE_DOWN;
        case VK_LEFT: return SDL_SCANCODE_LEFT;
        case VK_RIGHT: return SDL_SCANCODE_RIGHT;
        // Keys a game's own tools read (Burnout 3's HUD editing).
        case VK_PRIOR: return SDL_SCANCODE_PAGEUP;
        case VK_NEXT: return SDL_SCANCODE_PAGEDOWN;
        case VK_HOME: return SDL_SCANCODE_HOME;
        case VK_END: return SDL_SCANCODE_END;
        case VK_INSERT: return SDL_SCANCODE_INSERT;
        case VK_DELETE: return SDL_SCANCODE_DELETE;
        case VK_NUMPAD8: return SDL_SCANCODE_KP_8;
        case VK_NUMPAD4: return SDL_SCANCODE_KP_4;
        case VK_NUMPAD5: return SDL_SCANCODE_KP_5;
        case VK_NUMPAD6: return SDL_SCANCODE_KP_6;
        default: break;
    }
    if (key >= VK_F1 && key <= VK_F11) return SDL_SCANCODE_F1 + (key - VK_F1);
    if (key >= 'A' && key <= 'Z') return SDL_SCANCODE_A + (key - 'A');
    if (key >= '1' && key <= '9') return SDL_SCANCODE_1 + (key - '1');
    if (key == '0') return SDL_SCANCODE_0;
    return -1;
}
#endif

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM w, LPARAM l) {
    switch (message) {
        case WM_CLOSE:
            g_closed = true;
            DestroyWindow(window);
            return 0;
        case WM_DESTROY:
            g_window = nullptr;
            return 0;
        default:
            return DefWindowProcW(window, message, w, l);
    }
}

bool OpenGdiWindow(const wchar_t* title, int width, int height) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"ps2host";
    RegisterClassW(&wc);
    RECT r = {0, 0, width, height};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_window = CreateWindowW(L"ps2host", title, WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                             CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr,
                             wc.hInstance, nullptr);
    return g_window != nullptr;
}

void PresentGdi() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    if (!g_window) return;
    std::vector<u32> pixels;
    u32 w = 0, h = 0;
    if (!gs::TheGs().Picture(&pixels, &w, &h)) return;
    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = LONG(w);
    info.bmiHeader.biHeight = -LONG(h);                 // top-down rows
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    RECT client;
    GetClientRect(g_window, &client);
    HDC dc = GetDC(g_window);
    SetStretchBltMode(dc, HALFTONE);
    StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0, LONG(w), LONG(h), pixels.data(), &info,
                  DIB_RGB_COLORS, SRCCOPY);
    ReleaseDC(g_window, dc);
}

/* The "aspect" setting as the display takes it: "4:3" (the default), "16:9",
   or "auto" (0, the window's own shape). */
float SettingAspect() {
    const std::string aspect = Setting("aspect", "4:3");
    if (aspect == "16:9") return 16.0f / 9.0f;
    if (aspect == "auto") return 0.0f;
    return 4.0f / 3.0f;
}

}  // namespace

std::string g_title;                // the window's title as opened (SetWindowStatus adds to it)
void ApplyCursorArt();               // the game's pointer to a window opened after it was set

std::string& GameTitleText() {
    static std::string title;
    return title;
}

void SetGameTitle(const std::string& title) { GameTitleText() = title; }
const std::string& GameTitle() { return GameTitleText(); }

bool OpenWindow(const wchar_t* title, int width, int height) {
#ifdef PS2HOST_VULKAN
    std::string narrow;
    for (const wchar_t* p = title; *p; ++p) narrow += char(*p < 128 ? *p : '?');
    if (!GameTitle().empty()) narrow = GameTitle();
    g_title = narrow;
    std::string error;
    g_display = ps2gpu::CreateVulkanDisplay(narrow.c_str(), width, height, &error);
    if (g_display) {
        Note("window", "window: Vulkan on " + g_display->Describe());
        if (SettingOn("fullscreen")) g_display->SetFullscreen(true);
        ApplyCursorArt();
        g_display->SetPresentObserver([](std::uint64_t due_ns, std::uint64_t shown_ns, bool dropped) {
            // Shown on the display's own thread (its lane named so); dropped
            // on the thread handing it a newer one.
            thread_local bool named = false;
            if (!dropped && !named) {
                timeline::NameThread("present");
                named = true;
            }
            if (dropped) timeline::Dropped(due_ns);
            else timeline::Presented(due_ns, shown_ns);
        });
        if (SettingOn("vsync")) g_display->SetVsync(true);
        if (Setting("fullscreen_refresh", "match") == "desktop") g_display->SetGameRate(0);
        // present_no_wait (on by default): a picture whose copy would wait for the GPU
        // skipped, not waited for on the GS thread.
        g_display->SetPresentWait(Setting("present_no_wait", "on") != "on");
        g_display->SetAspect(SettingAspect());
        return true;
    }
    Note("window", "window: no Vulkan display (" + error + "); drawing with GDI");
#endif
    return OpenGdiWindow(title, width, height);
}

bool WindowOpen() {
#ifdef PS2HOST_VULKAN
    if (g_display) return !g_closed;
#endif
    return g_window != nullptr;
}

bool WindowFocused() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Focused();
#endif
    return g_window && GetForegroundWindow() == g_window;
}

bool WindowClosed() { return g_closed; }

bool HostKeyDown(int key) {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->KeyDown(Scancode(key));
#endif
    return (GetAsyncKeyState(key) & 0x8000) != 0;
}

void HostRumble(u8 large, u8 small) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetRumble(u16(large * 257), u16(small * 257));
#else
    (void)large, (void)small;
#endif
}

bool TakeHostMouse(HostMouseState* state) {
    if (ScriptedMouse(state)) return true;      // --mouse
#ifdef PS2HOST_VULKAN
    ps2gpu::Display::MouseState m;
    if (g_display && g_display->TakeMouse(&m)) {
        state->over = m.over;
        state->x = m.x;
        state->y = m.y;
        state->held = m.held;
        state->pressed = m.pressed;
        state->wheel = m.wheel;
        state->moved = m.moved;
        return true;
    }
#endif
    (void)state;
    return false;
}

// The game's pointer (SetHostCursor), kept for a window opened after it is set.
struct CursorArt {
    std::vector<u32> rgba;
    u32 width = 0, height = 0, hot_x = 0, hot_y = 0;
};
CursorArt& TheCursorArt() {
    static CursorArt art;
    return art;
}

void ApplyCursorArt() {
#ifdef PS2HOST_VULKAN
    const CursorArt& a = TheCursorArt();
    if (g_display && !a.rgba.empty())
        g_display->SetCursorArt(std::vector<std::uint32_t>(a.rgba.begin(), a.rgba.end()), a.width, a.height,
                                a.hot_x, a.hot_y);
#endif
}

void SetHostCursor(const u32* rgba, u32 width, u32 height, u32 hot_x, u32 hot_y) {
    TheCursorArt() = CursorArt{std::vector<u32>(rgba, rgba + std::size_t(width) * height), width, height, hot_x, hot_y};
    ApplyCursorArt();
}

void ShowHostCursor(bool on) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->ShowGameCursor(on);
#else
    (void)on;
#endif
}

bool HostPad(HostPadState* state) {
#ifdef PS2HOST_VULKAN
    ps2gpu::Display::PadState pad;
    if (g_display && g_display->Pad(&pad)) {
        state->buttons = pad.buttons;
        for (int i = 0; i < 4; ++i) state->sticks[i] = pad.sticks[i];
        state->l2 = pad.l2;
        state->r2 = pad.r2;
        return true;
    }
#endif
    (void)state;
    return false;   // the GDI window has none
}

namespace {
// --gpu-show: where the window's picture comes from instead of the software GS.
std::function<bool(std::vector<u32>*, u32*, u32*)> g_picture_source;
}  // namespace

#ifdef PS2HOST_VULKAN
ps2gpu::Gpu* WindowGpu() { return g_display ? g_display->SharedGpu() : nullptr; }
#endif

void SetPictureSource(std::function<bool(std::vector<u32>*, u32*, u32*)> source) {
    g_picture_source = std::move(source);
}

// Alt+Enter changes fullscreen in the display's Pump: saved as the setting.
void KeepFullscreenSetting() {
#ifdef PS2HOST_VULKAN
    static int last = -1;
    const int now = g_display && g_display->Fullscreen() ? 1 : 0;
    if (last >= 0 && now != last) SetSettingOn("fullscreen", now != 0);
    last = now;
#endif
}

bool PictureRate::Tick(bool fresh, u64 now_ns, std::string* line) {
    if (!from_ns) {                             // the first blank starts the clock
        from_ns = now_ns;
        return false;
    }
    ++blanks;
    if (fresh) ++pictures;
    const u64 span = now_ns - from_ns;
    if (span < 1000000000ull) return false;
    char text[96];
    std::snprintf(text, sizeof text, "%.1f images/s   %.1f VBlank/s", double(pictures) * 1e9 / double(span),
                  double(blanks) * 1e9 / double(span));
    *line = text;
    from_ns = now_ns;
    blanks = pictures = 0;
    return true;
}

namespace {
std::atomic<u64> g_fresh_pictures{0};
}  // namespace

void SetWindowGameRate(int hz) {
#ifdef PS2HOST_VULKAN
    // fullscreen_refresh=desktop: the desktop's refresh, not one the game's rate divides.
    static int last = 60;
    last = hz;
    if (g_display) g_display->SetGameRate(Setting("fullscreen_refresh", "match") == "desktop" ? 0 : last);
#else
    (void)hz;
#endif
}

void CountPresentedPicture(bool fresh) {
    if (fresh) g_fresh_pictures.fetch_add(1, std::memory_order_relaxed);
}

bool PumpWindow() {
#ifdef PS2HOST_VULKAN
    if (g_display) {
        if (!g_display->Pump()) g_closed = true;
        KeepFullscreenSetting();
        if (g_display->TakeCapture()) RequestCapture();
        // Tab: the texture pack on or off, said on the screen for two seconds.
        static std::chrono::steady_clock::time_point notice_until{};
        static bool notice = false;
        if (g_display->TakeTexturePackToggle() && texture_pack::Count()) {
            const bool on = !texture_pack::Enabled();
            texture_pack::SetEnabled(on);
            SetSettingOn("texture_pack", on);
            SetWindowOverlay({on ? "TEXTURE PACK ON  (Tab)" : "TEXTURE PACK OFF  (Tab)"});
            notice = true;
            notice_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        }
        if (notice && std::chrono::steady_clock::now() >= notice_until) {
            notice = false;
            SetWindowOverlay({});
        }
        // show_fps=on: the game's picture rate and the blanks', once a second.
        static const bool show_fps = SettingOn("show_fps") || std::getenv("PS2HOST_SHOW_FPS");
        if (show_fps && !notice) {
            static PictureRate rate;
            static u64 seen = 0;
            const u64 pictures = g_fresh_pictures.load(std::memory_order_relaxed);
            std::string line;
            const u64 now = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count());
            if (rate.Tick(pictures != seen, now, &line)) {
                SetWindowOverlay({line});
                std::fprintf(stderr, "[fps] %s\n", line.c_str());      // in the log too, to quote
            }
            seen = pictures;
        }
        return !g_closed;
    }
#endif
    if (g_window) PresentGdi();
    return true;
}

void SetWindowFullscreen(bool on) {
    SetSettingOn("fullscreen", on);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetFullscreen(on);
#endif
}

void SetWindowOverlay(const std::vector<std::string>& lines) {
#ifdef PS2HOST_VULKAN
    if (!g_display) return;
    if (lines.empty()) {
        g_display->SetOverlay({}, 0, 0);
        return;
    }
    // Text as Windows draws it, sized to the window (a 1080-line screen:
    // 22-pixel lines), white on a dark panel; a line starting with '>' in
    // yellow (the one chosen).
    const std::uint32_t screen = std::max(480u, g_display->PixelHeight());
    const int line = std::max(14, int(screen / 48));
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT font = CreateFontA(-(line - 4), 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS,
                             CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FIXED_PITCH | FF_MODERN, "Consolas");
    HGDIOBJ old_font = SelectObject(dc, font);
    int width = 0;
    for (const std::string& text : lines) {
        SIZE size{};
        GetTextExtentPoint32A(dc, text.c_str(), int(text.size()), &size);
        width = std::max(width, int(size.cx));
    }
    width = std::min(1280, width + line);
    const int height = std::min(720, int(lines.size()) * line + line / 2);
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof info.bmiHeader;
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    std::vector<std::uint32_t> pixels;
    if (bitmap && bits) {
        HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
        const RECT all{0, 0, width, height};
        HBRUSH panel = CreateSolidBrush(RGB(16, 20, 32));
        FillRect(dc, &all, panel);
        DeleteObject(panel);
        SetBkMode(dc, TRANSPARENT);
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const bool chosen = !lines[i].empty() && lines[i][0] == '>';
            SetTextColor(dc, chosen ? RGB(255, 210, 60) : i == 0 ? RGB(120, 190, 255) : RGB(235, 235, 235));
            TextOutA(dc, line / 2, int(i) * line + line / 4, lines[i].c_str(), int(lines[i].size()));
        }
        GdiFlush();
        pixels.assign(static_cast<const std::uint32_t*>(bits), static_cast<const std::uint32_t*>(bits) + width * height);
        for (std::uint32_t& p : pixels) p |= 0xFF000000u;
        SelectObject(dc, old_bitmap);
        DeleteObject(bitmap);
    }
    SelectObject(dc, old_font);
    DeleteObject(font);
    DeleteDC(dc);
    g_display->SetOverlay(std::move(pixels), std::uint32_t(width), std::uint32_t(height));
#else
    (void)lines;
#endif
}

void SetWindowStatus(const std::string& status) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetTitle(status.empty() ? g_title : g_title + "  -  " + status);
#else
    (void)status;
#endif
}

void SetWindowAspect(const std::string& aspect) {
    SetSetting("aspect", aspect);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetAspect(SettingAspect());
#endif
}

float PictureAspect() {
    const float aspect = SettingAspect();
    if (aspect > 0.0f) return aspect;
#ifdef PS2HOST_VULKAN
    if (g_display) {
        const float window = g_display->WindowAspect();
        if (window > 0.0f) return window;
    }
#endif
    return 4.0f / 3.0f;
}

void SetWindowVsync(bool on) {
    SetSettingOn("vsync", on);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetVsync(on);
#endif
}

bool WindowVsync() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Vsync();
#endif
    return SettingOn("vsync");
}

bool WindowFullscreen() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Fullscreen();
#endif
    return SettingOn("fullscreen");
}

ps2gpu::Display* WindowDisplay() {
#ifdef PS2HOST_VULKAN
    return g_display.get();
#else
    return nullptr;
#endif
}

void PresentFrame() {
#ifdef PS2HOST_VULKAN
    if (g_display) {
        if (!g_display->Pump()) {
            g_closed = true;
            return;
        }
        KeepFullscreenSetting();
        std::vector<u32> pixels;
        u32 w = 0, h = 0;
        const bool got = (g_picture_source && g_picture_source(&pixels, &w, &h)) || gs::TheGs().Picture(&pixels, &w, &h);
        if (got && !g_display->Present(pixels.data(), w, h)) g_closed = true;
        return;
    }
#endif
    if (g_window) PresentGdi();
}

}  // namespace ps2host
