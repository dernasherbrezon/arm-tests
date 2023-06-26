# About

Various tests for DSP algorithms using ARM intristics

# Build

Based on the available toolchain some combinations of compilation flags and NEON implementations might fail to build. This is expected. Just run with "-i -k":

```bash
mkdir build
cd build
cmake ..
make -i -k
```

# Run

Make will create lots of small executable binaries built with different compilation flags and optimizations.

```run_types.sh``` script can be used to prepare results in json format for future visualizations.

