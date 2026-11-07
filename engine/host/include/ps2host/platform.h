/*
 * What the host takes from Win32 and the Microsoft C library, elsewhere: the
 * keys HostKeyDown takes (run.h) are Win32's virtual keys - windows.h's where
 * there is one, the same values elsewhere (letters and digits their capital
 * ASCII, as in Win32) - and the C library's _s time functions.
 */

#ifndef PS2HOST_PLATFORM_H
#define PS2HOST_PLATFORM_H

#ifndef _WIN32
#include <ctime>

enum : int {
    VK_BACK = 0x08,
    VK_RETURN = 0x0D,
    VK_PRIOR = 0x21,
    VK_NEXT = 0x22,
    VK_END = 0x23,
    VK_HOME = 0x24,
    VK_LEFT = 0x25,
    VK_UP = 0x26,
    VK_RIGHT = 0x27,
    VK_DOWN = 0x28,
    VK_INSERT = 0x2D,
    VK_DELETE = 0x2E,
    VK_NUMPAD4 = 0x64,
    VK_NUMPAD5 = 0x65,
    VK_NUMPAD6 = 0x66,
    VK_NUMPAD8 = 0x68,
    VK_F1 = 0x70,
    VK_F11 = 0x7A,
};

inline int localtime_s(std::tm* out, const std::time_t* time) { return localtime_r(time, out) ? 0 : 1; }
inline int gmtime_s(std::tm* out, const std::time_t* time) { return gmtime_r(time, out) ? 0 : 1; }
#endif

#endif  // PS2HOST_PLATFORM_H
