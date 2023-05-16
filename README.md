# Ray tracing with CUDA
Ray tracer that uses GPGPU.

## Setting up
Repository doesn't contain any project files.
They can be created with srcipts/Setup.bat which is set for VisualStudio 2019.
Building project files (.sln, .vcxproj) achived with premake.

VS configuration:
- VS 2019,
- C++ 17,

Current CUDA configuration:
- CUDA Toolkit - 12.1,
- Default compute capability - I guess it inherits on its own (mine is 5.2).

### Note
Running scripts/Setup.bat will result in generating new .sln file and configuration in .vcxproj with all CUDA settings.

### Framework for GUI
- [Walnut](https://github.com/TheCherno/Walnut)