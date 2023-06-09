# Ray tracing with CUDA
# What is ray tracing?
Ray tracing is a rendering technique that allows to simulate realistic behaviour of light in games, animations etc. It also allows to perform shades, reflections and lighting.

# Project requirements:
- Visual Studio 2019
- Vulkan SDK 1.3
- CUDA toolkit 12.1
- Compute Capability >=5.2

# Build / compilation instruction
Having downloaded project, launch Setup.bat file in order to configure CUDA in the project and create solution file.
```
├── bin
├── doc
├── source
│   ├── scripts
│    	└── Setup.bat
```
Then you can just compile it using MSVC compiler.

<sub>Used framework GUI</sub> 
[Walnut](https://github.com/TheCherno/Walnut)