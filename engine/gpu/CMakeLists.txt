# The port's display on the GPU: SDL3 for the window, input and events, Vulkan
# for the picture. Vulkan reaches every platform the port aims at - Windows,
# Linux and Android natively, macOS through MoltenVK - and gives the control a GS
# renderer needs; SDL3 hides each platform's window and input. Both come from
# their releases at configure time (FetchContent), so nothing is installed on
# the machine and nothing of theirs is committed.
#
# Stage one presents the software GS's picture (a copy and a scaled blit, no
# shaders). The GPU GS renderer builds on this device and swapchain.

cmake_minimum_required(VERSION 3.20)
project(ps2gpu CXX C)

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.363
    GIT_SHALLOW TRUE)
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE)
FetchContent_Declare(sdl3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.4.18
    GIT_SHALLOW TRUE)

set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vulkan_headers)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(volk sdl3)

# Shaders: GLSL compiled to SPIR-V at build time by glslang, from its release
# like the others, and embedded as arrays (glslang --vn).
FetchContent_Declare(glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glslang)

set(PS2GPU_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${PS2GPU_SHADER_DIR})
set(PS2GPU_SHADER_HEADERS)
foreach(shader gs.vert gs.frag date.vert date.frag swizzle.comp decode.comp depth_load.frag patch.comp)
    string(REPLACE "." "_" name ${shader})
    set(out ${PS2GPU_SHADER_DIR}/${name}.h)
    add_custom_command(OUTPUT ${out}
        COMMAND glslang-standalone -V --target-env vulkan1.3 --vn spirv_${name}
                -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/${shader}
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/${shader} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
                glslang-standalone
        COMMENT "glslang ${shader}")
    list(APPEND PS2GPU_SHADER_HEADERS ${out})
endforeach()
# The fragment shader without its second output, for devices without
# dual-source blending.
set(out ${PS2GPU_SHADER_DIR}/gs_frag_single.h)
add_custom_command(OUTPUT ${out}
    COMMAND glslang-standalone -V --target-env vulkan1.3 -DSINGLE_OUTPUT --vn spirv_gs_frag_single
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag glslang-standalone
    COMMENT "glslang gs.frag (single output)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# The swizzle straight to and from a colour target (a storage image).
set(out ${PS2GPU_SHADER_DIR}/swizzle_image_comp.h)
add_custom_command(OUTPUT ${out}
    COMMAND glslang-standalone -V --target-env vulkan1.3 -DIMAGE --vn spirv_swizzle_image_comp
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
            glslang-standalone
    COMMENT "glslang swizzle.comp (a target's image)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# ...and from a depth buffer, sampled.
set(out ${PS2GPU_SHADER_DIR}/swizzle_depth_comp.h)
add_custom_command(OUTPUT ${out}
    COMMAND glslang-standalone -V --target-env vulkan1.3 -DDEPTH_IMAGE --vn spirv_swizzle_depth_comp
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
            glslang-standalone
    COMMENT "glslang swizzle.comp (a depth buffer)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# ...and blending in the shader, from a copy of the target (a factor over 1).
set(out ${PS2GPU_SHADER_DIR}/gs_frag_blend.h)
add_custom_command(OUTPUT ${out}
    COMMAND glslang-standalone -V --target-env vulkan1.3 -DSHADER_BLEND --vn spirv_gs_frag_blend
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag glslang-standalone
    COMMENT "glslang gs.frag (blending in the shader)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
add_custom_target(ps2gpu_shaders DEPENDS ${PS2GPU_SHADER_HEADERS})

add_library(ps2gpu STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/vulkan_display.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gs_renderer.cpp)
add_dependencies(ps2gpu ps2gpu_shaders)
target_include_directories(ps2gpu PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include PRIVATE ${PS2GPU_SHADER_DIR})
target_compile_features(ps2gpu PUBLIC cxx_std_17)
target_link_libraries(ps2gpu PUBLIC SDL3::SDL3-static PRIVATE volk::volk Vulkan::Headers)
if(MSVC)
    target_compile_options(ps2gpu PRIVATE /W3)
endif()

add_executable(ps2gpu_smoke ${CMAKE_CURRENT_LIST_DIR}/tests/display_smoke.cpp)
target_link_libraries(ps2gpu_smoke PRIVATE ps2gpu)
add_executable(ps2gpu_tests ${CMAKE_CURRENT_LIST_DIR}/tests/renderer_tests.cpp)
target_link_libraries(ps2gpu_tests PRIVATE ps2gpu)
