CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(comp_prog_cnn_proj C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.c")

add_executable(train ${SRC_FILES})
target_include_directories(train PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(train PRIVATE -Wall -Wextra -g)
target_link_libraries(train PRIVATE m)

# Place runtime binary in the top-level bin/ directory
set_target_properties(train PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

# Optionally use stb_image for real image loading. If enabled, place
# `stb_image.h` in `third_party/` and configure with -DUSE_STB=ON.
option(USE_STB "Enable stb_image-based image loading (requires third_party/stb_image.h)" OFF)
if(USE_STB)
    target_compile_definitions(train PRIVATE HAVE_STB_IMAGE=1)
    target_include_directories(train PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
endif()
