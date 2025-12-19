cmake_minimum_required(VERSION 3.0)
project(odinann)

set(CMAKE_STANDARD 17)
option(USE_AIO "Use AIO instead of liburing" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

# Check if liburing is installed in a custom location
if(EXISTS ${PROJECT_SOURCE_DIR}/third_party/liburing/install)
    set(LIBURING_CUSTOM_INSTALL TRUE)
    set(LIBURING_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/liburing/install/include)
    set(LIBURING_LIB_DIR ${PROJECT_SOURCE_DIR}/third_party/liburing/install/lib)
endif()

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
	${LIBURING_INCLUDE_DIR}
)

add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
)
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops)

message("Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	-DNDEBUG
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	# -DDIRECT_READ_CC # disable opt #1.1
	# -DBG_IO_THREAD # enable opt #1.2
	# -DDELTA_PRUNING # enable opt #2

	# -DNO_POLLUTE_ORIGINAL

	# optimizations.
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # starling: must disable this!
	-DOVERLAP_INIT
	-DDYN_PIPE_WIDTH
	
	# policy.
	# -DSTATIC_POLICY

	# unoptimizations.
	# -DNAIVE_PIPE
)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)
link_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src
	${LIBURING_LIB_DIR}
)

if(USE_AIO)
	message(STATUS "Using AIO for asynchronous I/O, compatible but slower.")
	link_libraries(aio)
	add_definitions(-DUSE_AIO)
else()
	message(STATUS "Using liburing for asynchronous I/O, faster but requires kernel support.")
	if(LIBURING_CUSTOM_INSTALL)
		link_libraries(${LIBURING_LIB_DIR}/liburing.a)
	else()
		link_libraries(uring)
	endif()
endif()

link_libraries(
	jemalloc
)
# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)
# add_compile_options(-fsanitize=undefined -lubsan)
# add_link_options(-fsanitize=undefined)

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
try_run(
	RUN_RESULT
	COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp"
	COMPILE_OUTPUT_VARIABLE COMPILE_OUT
	RUN_OUTPUT_VARIABLE RUN_OUT
)

if(COMPILE_RESULT AND RUN_RESULT EQUAL 3) # AVX512 + AVX2
	set(CPU_SUPPORTS_AVX512 TRUE)
	message(STATUS "CPU supports AVX512")
elseif(COMPILE_RESULT AND RUN_RESULT EQUAL 1) # AVX2
	set(CPU_SUPPORTS_AVX2 TRUE)
	message(STATUS "CPU supports AVX2")
else()
	message(STATUS "CPU doesn't support AVX512 or AVX2")
endif()

if(CPU_SUPPORTS_AVX512)
	add_definitions(-mavx512f)
elseif(CPU_SUPPORTS_AVX2)
	add_definitions(-mavx2)
endif()

file(GLOB_RECURSE SOURCES 
	"src/*.cpp"
	"src/*.c"
	"src/*/*.cpp"
	"src/*/*.c"
)
add_library(odinann STATIC ${SOURCES})

add_executable(search_memory_index tests/search_memory_index.cpp)
target_link_libraries(search_memory_index odinann)

add_executable(build_memory_index tests/build_memory_index.cpp)
target_link_libraries(build_memory_index odinann)

add_executable(search_disk_index tests/search_disk_index.cpp)
target_link_libraries(search_disk_index odinann)

add_executable(build_disk_index tests/build_disk_index.cpp)
target_link_libraries(build_disk_index odinann)

add_executable(update_disk_index tests/update_disk_index.cpp)
target_link_libraries(update_disk_index odinann)

add_executable(search_cache_index tests/search_cache_index.cpp)
target_link_libraries(search_cache_index odinann)

add_executable(build_cache_index tests/build_cache_index.cpp)
target_link_libraries(build_cache_index odinann)