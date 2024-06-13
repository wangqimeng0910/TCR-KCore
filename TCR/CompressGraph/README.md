# CompressGraph

## Description
Some data processing modules have been added to the [Compress open source code](https://github.com/ZhengChenCS/CompressGraph)

See what each **cpp file** does in the readme file under the /src directory


## System Dependency
 - [CMake](https://gitlab.kitware.com/cmake/cmake)
 - Optional(CPU): [Ligra](https://github.com/jshun/ligra.git)
 - Optional(GPU): [Gunrock](https://github.com/gunrock/gunrock.git)

## Compilation
```bash
cd CompressGraph
mkdir -p build
cd build && cmake .. && make -j



