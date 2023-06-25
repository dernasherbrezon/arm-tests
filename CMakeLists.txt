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

set(map_TEST_PREFETCH pref)
set(map_TEST_NOPREFETCH npref)

foreach(TEST_TYPE TEST_GENERIC TEST_NEON4Q TEST_NEON2Q TEST_NEON1Q)
    foreach(TEST_MEMORY TEST_ALIGN_MEMORY TEST_NONALIGN_MEMORY)
        foreach(TEST_SIZE TEST_ALIGN_SIZE TEST_NONALIGN_SIZE)
            foreach(TEST_FETCH TEST_PREFETCH TEST_NOPREFETCH)
                set(TEST_TYPE_HUMAN_FRIENDLY ${map_${TEST_TYPE}})
                set(TEST_MEMORY_HUMAN_FRIENDLY ${map_${TEST_MEMORY}})
                set(TEST_SIZE_HUMAN_FRIENDLY ${map_${TEST_SIZE}})
                set(TEST_FETCH_HUMAN_FRIENDLY ${map_${TEST_FETCH}})
                set(EXEC_SUFFIX "${TEST_TYPE_HUMAN_FRIENDLY}_${TEST_MEMORY_HUMAN_FRIENDLY}_${TEST_SIZE_HUMAN_FRIENDLY}_${TEST_FETCH_HUMAN_FRIENDLY}")
                set(TEST_ARGUMENTS "-D${TEST_TYPE} -D${TEST_MEMORY} -D${TEST_SIZE} -D${TEST_FETCH}")

                add_executable(dot_prod_${EXEC_SUFFIX} dot_prod.c)
                target_compile_options(dot_prod_${EXEC_SUFFIX} PUBLIC ${TEST_ARGUMENTS} -mfpu=neon)
                add_executable(dot_prod_${EXEC_SUFFIX}_optimized_64 dot_prod.c)
                target_compile_options(dot_prod_${EXEC_SUFFIX}_optimized_64 PUBLIC ${TEST_ARGUMENTS} -O3 -march=armv8-a -mfpu=neon)
                add_executable(dot_prod_${EXEC_SUFFIX}_optimized_32 dot_prod.c)
                target_compile_options(dot_prod_${EXEC_SUFFIX}_optimized_32 PUBLIC ${TEST_ARGUMENTS} -O3 -mfloat-abi=hard -march=armv7-a -mfpu=neon)
            endforeach()
        endforeach()
    endforeach()
endforeach()



