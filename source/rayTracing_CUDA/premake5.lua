require("../premake5-cuda")

project "rayTracing_CUDA"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   files {"src/app/**.h", "src/app/**.cpp"}

   -- CUDA configurations
    buildcustomizations "BuildCustomizations/CUDA 11.8"

    externalwarnings "Off" -- thrust gives a lot of warnings
    cudaFiles { "src/scene/**.cu", "src/scene/**.cpp", "src/scene/**.h" } -- files to be compiled into binaries
    -- cudaKeep "On" -- keep temporary output files
    -- cudaFastMath "On"
    cudaRelocatableCode "On"
    -- cudaVerbosePTXAS "On"
    -- cudaMaxRegCount "32"

   includedirs
   {
      "../vendor/imgui",
      "../vendor/glfw/include",

      "../Walnut/src",

      "%{IncludeDir.VulkanSDK}",
      "%{IncludeDir.glm}",
   }

    links
    {
        "Walnut"
    }

   targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

   filter "system:windows"
      systemversion "latest"
      defines { "WL_PLATFORM_WINDOWS" }

   filter "configurations:Debug"
      defines { "WL_DEBUG" }
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      defines { "WL_RELEASE" }
      runtime "Release"
      optimize "On"
      symbols "On"

   filter "configurations:Dist"
      kind "WindowedApp"
      defines { "WL_DIST" }
      runtime "Release"
      optimize "On"
      symbols "Off"