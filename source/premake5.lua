-- premake5.lua
workspace "rayTracing_CUDA"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "rayTracing_CUDA"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "WalnutExternal.lua"
include "rayTracing_CUDA"