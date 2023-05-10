# Ray tracing with CUDA
Ray tracer. In development.

## Setting up
Repository doesn't contain any project files.
They can be created with srcipts/Setup.bat which is set for VisualStudio 2019.
Building .sln, .vcxproj achived with premake.

VS configuration:
- VS 2019,
- C++ 17,

Current CUDA configuration:
- CUDA Toolkit - 11.8
- Default compute capability - 5.2

### Note
Running scripts/Setup.bat will result in generating new .sln file without CUDA configuration. 

### Framework for GUI
- [Walnut](https://github.com/TheCherno/Walnut)