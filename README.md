# Ray tracing with CUDA
Ray tracer. In development.

## Setting up
Repository contains:
- .sln text file which has configuration for grouping projects.
- .vcxproj (XML file) used to build project using MSBUILD it defines how to compile and link files.

VS configuration:
- VS 2019,
- C++ 17,
- x64 platform.

Current CUDA configuration:
- CUDA Toolkit - 11.8
- Default compute capability - 5.2

### Note
Running scripts/Setup.bat will result in generating new .sln file without CUDA configuration. 

### Framework for GUI
- [Walnut](https://github.com/TheCherno/Walnut)