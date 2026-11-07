# Stuntman, native. Build with engine/tools/port.py, which generates the
# dispatch table and drives this file. (From engine/templates/game.)

cmake_minimum_required(VERSION 3.20)
project(stuntman_port LANGUAGES CXX)
if(WIN32)
    enable_language(RC)                     # the icon and version
endif()

set(PS2_GENERATED_DIR "" CACHE PATH "dispatch.cpp, from engine/tools/port.py")

# The C and C++ runtime linked in (/MT): the executable runs on a Windows
# without the Visual C++ redistributable - every library here built so too.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# Source files' names (__FILE__) from the repository's root, not the folder it
# was built in: a release names no one's folders.
if(MSVC)
    get_filename_component(repo_root ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
    file(TO_NATIVE_PATH "${repo_root}/" repo_root_native)
    add_compile_options("/d1trimfile:${repo_root_native}")
else()
    # GCC and Clang: the same (the build folder, with the libraries' sources,
    # is in the repository); -O2 as MSVC's.
    get_filename_component(repo_root ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
    add_compile_options("-ffile-prefix-map=${repo_root}/=")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../engine/host ${CMAKE_BINARY_DIR}/ps2host)

ps2_add_port(stuntman
    GAME ${CMAKE_CURRENT_LIST_DIR}/..
    GENERATED ${PS2_GENERATED_DIR}
    TRANSLATED ${CMAKE_CURRENT_LIST_DIR}/../src/lifted
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/overrides.cpp)
