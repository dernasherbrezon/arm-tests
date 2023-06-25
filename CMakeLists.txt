cmake_minimum_required(VERSION 3.18)
project(arm_tests C)

set(CMAKE_C_STANDARD 11)

set(map_TEST_GENERIC generic)
set(map_TEST_NEON4Q neon4q)
set(map_TEST_NEON2Q neon2q)
set(map_TEST_NEON1Q neon1q)

set(map_TEST_ALIGN_MEMORY am)
set(map_TEST_NONALIGN_MEMORY nam)

set(map_TEST_ALIGN_SIZE as)
set(map_TEST_NONALIGN_SIZE nas)

foreach(TEST_TYPE TEST_GENERIC TEST_NEON4Q TEST_NEON2Q TEST_NEON1Q)
    foreach(TEST_MEMORY TEST_ALIGN_MEMORY TEST_NONALIGN_MEMORY)
        foreach(TEST_SIZE TEST_ALIGN_SIZE TEST_NONALIGN_SIZE)
            set(TEST_TYPE_HUMAN_FRIENDLY ${map_${TEST_TYPE}})
            set(TEST_MEMORY_HUMAN_FRIENDLY ${map_${TEST_MEMORY}})
            set(TEST_SIZE_HUMAN_FRIENDLY ${map_${TEST_SIZE}})
            set(EXEC_SUFFIX "${TEST_TYPE_HUMAN_FRIENDLY}_${TEST_MEMORY_HUMAN_FRIENDLY}_${TEST_SIZE_HUMAN_FRIENDLY}")

            add_executable(dot_prod_${EXEC_SUFFIX} dot_prod.c)
            target_compile_options(dot_prod_${EXEC_SUFFIX} PUBLIC -D${TEST_TYPE} -D${TEST_MEMORY} -D${TEST_SIZE})
            add_executable(dot_prod_${EXEC_SUFFIX}_optimized_64 dot_prod.c)
            target_compile_options(dot_prod_${EXEC_SUFFIX}_optimized_64 PUBLIC -D${TEST_TYPE} -D${TEST_MEMORY} -D${TEST_SIZE} -O2 -march=armv8-a)
            add_executable(dot_prod_${EXEC_SUFFIX}_optimized_32 dot_prod.c)
            target_compile_options(dot_prod_${EXEC_SUFFIX}_optimized_32 PUBLIC -D${TEST_TYPE} -D${TEST_MEMORY} -D${TEST_SIZE} -O2 -mfloat-abi=hard -march=armv7-a)
        endforeach()
    endforeach()
endforeach()



