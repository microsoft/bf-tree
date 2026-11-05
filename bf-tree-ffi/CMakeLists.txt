cmake_minimum_required(VERSION 3.20)
project(BfTreeFFI LANGUAGES C CXX)

find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_package(Threads REQUIRED)

if(MSVC)
    if(CMAKE_GENERATOR_PLATFORM)
        set(BF_TREE_ARCHITECTURE "${CMAKE_GENERATOR_PLATFORM}")
    elseif(CMAKE_VS_PLATFORM_NAME)
        set(BF_TREE_ARCHITECTURE "${CMAKE_VS_PLATFORM_NAME}")
    else()
        set(BF_TREE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(BF_TREE_ARCHITECTURE MATCHES "^(ARM64|arm64|aarch64)$")
        set(BF_TREE_RUST_TARGET aarch64-pc-windows-msvc)
    elseif(BF_TREE_ARCHITECTURE MATCHES "^(x64|X64|x86_64|AMD64|amd64)$")
        set(BF_TREE_RUST_TARGET x86_64-pc-windows-msvc)
    else()
        message(FATAL_ERROR "Unsupported MSVC architecture: ${BF_TREE_ARCHITECTURE}")
    endif()
    set(BF_TREE_LIBRARY_NAME bf_tree_ffi.lib)
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(BF_TREE_RUST_TARGET aarch64-unknown-linux-gnu)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(BF_TREE_RUST_TARGET x86_64-unknown-linux-gnu)
    else()
        message(FATAL_ERROR "Unsupported Linux architecture")
    endif()
    set(BF_TREE_LIBRARY_NAME libbf_tree_ffi.a)
else()
    message(FATAL_ERROR "This C consumer build supports Windows MSVC and Linux")
endif()

set(BF_TREE_CARGO_TARGET "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(BF_TREE_DEBUG_LIBRARY
    "${BF_TREE_CARGO_TARGET}/${BF_TREE_RUST_TARGET}/debug/${BF_TREE_LIBRARY_NAME}")
set(BF_TREE_RELEASE_LIBRARY
    "${BF_TREE_CARGO_TARGET}/${BF_TREE_RUST_TARGET}/release/${BF_TREE_LIBRARY_NAME}")

add_custom_target(bf_tree_ffi_build
    COMMAND "${CARGO_EXECUTABLE}" +1.93.0 build
        --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
        --locked --target "${BF_TREE_RUST_TARGET}"
        --target-dir "${BF_TREE_CARGO_TARGET}"
        --profile $<IF:$<CONFIG:Debug>,dev,release>
    BYPRODUCTS "${BF_TREE_DEBUG_LIBRARY}" "${BF_TREE_RELEASE_LIBRARY}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM)

add_library(bf_tree_ffi STATIC IMPORTED GLOBAL)
set_target_properties(bf_tree_ffi PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_LOCATION "${BF_TREE_RELEASE_LIBRARY}"
    IMPORTED_LOCATION_DEBUG "${BF_TREE_DEBUG_LIBRARY}"
    IMPORTED_LOCATION_RELEASE "${BF_TREE_RELEASE_LIBRARY}"
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE
    MAP_IMPORTED_CONFIG_MINSIZEREL RELEASE
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_dependencies(bf_tree_ffi bf_tree_ffi_build)
target_link_libraries(bf_tree_ffi INTERFACE Threads::Threads)
if(MSVC)
    target_link_libraries(bf_tree_ffi INTERFACE advapi32 bcrypt ntdll userenv ws2_32)
else()
    target_link_libraries(bf_tree_ffi INTERFACE "${CMAKE_DL_LIBS}" m)
endif()

enable_testing()
add_executable(bf_tree_c_test tests/c_api.c)
set_target_properties(bf_tree_c_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(bf_tree_c_test PRIVATE bf_tree_ffi)
add_test(NAME BfTreeCAbi COMMAND bf_tree_c_test)

add_executable(bf_tree_cpp_test tests/cpp_api.cpp)
set_target_properties(bf_tree_cpp_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(bf_tree_cpp_test PRIVATE bf_tree_ffi)
add_test(NAME BfTreeCppAbi COMMAND bf_tree_cpp_test)

install(FILES include/bf_tree.h DESTINATION include)
install(FILES "$<TARGET_FILE:bf_tree_ffi>" DESTINATION lib)
install(FILES LICENSE.txt README.md DESTINATION share/bf-tree-ffi)
