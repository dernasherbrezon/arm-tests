#!/bin/bash

# This script can be used to create json-based datasets for visualizations

PREFIX=$1

echo "types"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_generic_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon1q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon2q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon4q_nam_nas_npref_optimized_64),"

echo "generic optimizations"
echo "\"${PREFIX}dot_prod_generic_nam_nas_npref\": $(./build/dot_prod_generic_nam_nas_npref),"
echo "\"${PREFIX}dot_prod_generic_nam_nas_npref\": $(./build/dot_prod_generic_nam_nas_npref_optimized_32),"
echo "\"${PREFIX}dot_prod_generic_nam_nas_npref\": $(./build/dot_prod_generic_nam_nas_npref_optimized_64),"

echo "neon2q optimizations"
echo "\"${PREFIX}dot_prod_neon2q_nam_nas_npref\": $(./build/dot_prod_neon2q_nam_nas_npref),"
echo "\"${PREFIX}dot_prod_neon2q_nam_nas_npref\": $(./build/dot_prod_neon2q_nam_nas_npref_optimized_32),"
echo "\"${PREFIX}dot_prod_neon2q_nam_nas_npref\": $(./build/dot_prod_neon2q_nam_nas_npref_optimized_64),"

echo "prefetch"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_generic_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_pref_optimized_64\": $(./build/dot_prod_generic_nam_nas_pref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon1q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_pref_optimized_64\": $(./build/dot_prod_neon1q_nam_nas_pref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon2q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_pref_optimized_64\": $(./build/dot_prod_neon2q_nam_nas_pref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon4q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_pref_optimized_64\": $(./build/dot_prod_neon4q_nam_nas_pref_optimized_64),"

echo "size align"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_generic_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_as_npref_optimized_64\": $(./build/dot_prod_generic_nam_as_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon1q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_as_npref_optimized_64\": $(./build/dot_prod_neon1q_nam_as_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon2q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_as_npref_optimized_64\": $(./build/dot_prod_neon2q_nam_as_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon4q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_nam_as_npref_optimized_64\": $(./build/dot_prod_neon4q_nam_as_npref_optimized_64),"

echo "memory align"
echo "\"${PREFIX}dot_prod_generic_nam_nas_npref_optimized_64\": $(./build/dot_prod_generic_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_generic_am_nas_npref_optimized_64\": $(./build/dot_prod_generic_am_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_neon1q_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon1q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_neon1q_am_nas_npref_optimized_64\": $(./build/dot_prod_neon1q_am_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_neon2q_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon2q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_neon2q_am_nas_npref_optimized_64\": $(./build/dot_prod_neon2q_am_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_neon4q_nam_nas_npref_optimized_64\": $(./build/dot_prod_neon4q_nam_nas_npref_optimized_64),"
echo "\"${PREFIX}dot_prod_neon4q_am_nas_npref_optimized_64\": $(./build/dot_prod_neon4q_am_nas_npref_optimized_64),"
