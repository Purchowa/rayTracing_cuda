# Ray tracing with CUDA
## What is ray tracing?
Ray tracing is a rendering technique that allows to simulate realistic behaviour of light in games, animations etc. It also allows to perform shades, reflections and lighting.

# Project requirements:
VS configuration:
- VS 2019,
- CUDA Toolkit - 12.1,
- Default compute capability - I guess it inherits on its own,
- Vulcan SDK.

# Build / compilation instruction
Having downloaded project, launch Setup.bat file in order to configure CUDA in the project and create solution file.
Building project files (.sln, .vcxproj) achived with premake.
```
├── bin
├── doc
├── source
│   ├── scripts
│    	└── Setup.bat
```

### Framework for GUI
- [Walnut](https://github.com/TheCherno/Walnut)