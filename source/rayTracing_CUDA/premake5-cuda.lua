require("cuda_premake_src/cuda-exported-variables")

if os.target() == "windows" then
    dofile("cuda_premake_src/premake5-cuda-vs.lua")
end
