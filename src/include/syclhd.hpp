/**
 * @file syclhd.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief Main header file for SYCL-HD Library
 * @version 0.1
 * @date 2024-02-01
 * @copyright GPL3
 * 
 */

#ifndef SYCLHD_HPP
#define SYCLHD_HPP

/// Include Core Library
#include "../core/model.hpp"

#ifndef HDC_PROBLEM_HPP
#error Problem not defined!
#endif

/// Include MAP Module
#include "../modules/map/map.hpp"

#ifndef HDC_MAP_HPP
#error MAP Model not defined!
#endif

/// Include BSC Module
#include "../modules/bsc/bsc.hpp"

#ifndef HDC_BSC_HPP
#error BSC Model not defined!
#endif

/// Include MAP Bipolar Module
#include "../modules/tmap/tmap.hpp"

#ifndef HDC_TMAP_HPP
#error MAPBP Model not defined!
#endif

/// Include LSE Classification module
#include "../modules/lse_classifier/lse.hpp"

#ifndef HDC_LSE_HPP
#error LSE Classifier not defined!
#endif

#endif //SYCLHD_HPP
