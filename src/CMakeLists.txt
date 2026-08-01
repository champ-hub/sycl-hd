# CORE SOURCES
set(syclhd_core_headers 
    include/syclhd.hpp
    core/utilities.hpp
    core/definitions.hpp
    core/model.hpp
    core/matrix.hpp
    core/macros.hpp
    core/query.hpp
    core/training.hpp
    core/rbf_enc.hpp
)

set(syclhd_core_source  
    core/query.cpp
    core/training.cpp
    core/rbf_enc.cpp
    core/utilities.cpp
)

#MODULES SOURCES
set(syclhd_modules_headers 
    modules/bsc/b8_char.hpp
    modules/tmap/t4_char.hpp
    modules/bsc/bsc.hpp
    modules/tmap/tmap.hpp
    modules/map/map.hpp
)

set(syclhd_modules_source  
    modules/bsc/bsc_encoders.cpp
    modules/bsc/bsc_query.cpp
    modules/bsc/bsc_vector_generators.cpp
    modules/bsc/bsc_training.cpp
    modules/tmap/tmap_encoders.cpp
    modules/tmap/tmap_query.cpp
    modules/tmap/tmap_vector_generators.cpp
    modules/tmap/tmap_training.cpp
    modules/map/map_encoders.cpp
    modules/map/map_query.cpp
    modules/map/map_vector_generators.cpp
    modules/map/map_training.cpp
)

set(sycl_hd_lse_headers
    modules/lse_classifier/lse.hpp
)

set(sycl_hd_lse_source
    modules/lse_classifier/lse.cpp
)

add_library(syclhd STATIC 
    ${syclhd_core_source}
    ${syclhd_core_headers}
    ${syclhd_modules_source}
    ${syclhd_modules_headers}
    ${sycl_hd_lse_source}
    ${sycl_hd_lse_headers}
)

target_link_libraries(syclhd PRIVATE openblas LAPACK::LAPACK)
target_link_directories(syclhd PRIVATE ${OpenBLAS_LIBRARY_DIRS} ${LAPACK_LIBRARY_DIRS})

add_sycl_to_target(TARGET syclhd SOURCES 
    ${syclhd_core_source} 
    ${syclhd_core_headers}
    ${syclhd_modules_source}
    ${syclhd_modules_headers}
    ${sycl_hd_lse_source}
    ${sycl_hd_lse_headers}
)

target_include_directories(syclhd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # Used for builds
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>                            # Used for installs
)

set_target_properties(syclhd PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER syclhd.hpp
)

include(CMakePackageConfigHelpers)

install(TARGETS syclhd
    EXPORT sycl-hd-config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

export(TARGETS syclhd
    FILE sycl-hd-config.cmake)

install(EXPORT
    sycl-hd-config
    DESTINATION "${CMAKE_INSTALL_DATADIR}/SYCL-HD/cmake"
)

