#pragma once
#include "aux_utils.h"
#include <string>

namespace pipeann {

// Enhanced build function with structured configuration
template<typename T, typename TagT = uint32_t>
BuildResult build_disk_index_from_config(const DiskIndexBuildConfig& config, Metric _compareMetric);

// Enhanced build function with detailed result reporting
template<typename T, typename TagT = uint32_t>
BuildResult build_disk_index_with_result(const char* dataPath, const char* indexFilePath, 
                                   uint32_t R, uint32_t L, uint32_t M,
                                   uint32_t num_threads, uint32_t PQ_bytes,
                                   Metric _compareMetric, bool single_file_index,
                                   const char* tag_file = nullptr);

}  // namespace pipeann