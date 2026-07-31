add_executable(dofmex dofmex.cpp)
#target_link_libraries(dofmex PRIVATE dphdc)
target_link_libraries(dofmex PRIVATE syclhd)
add_sycl_to_target(TARGET dofmex SOURCES dofmex.cpp)