# CUDA 3D Hydraulic Erosion Simulation with layered stacks
This repository contains the full source code of our VMV 2024 paper **3D Real-Time Hydraulic Erosion Simulation using Multi-Layered Heightmaps**

# Setup
We provided a Visual Studio solution file. We have only tested our code on Windows, but it should be possible to get it to work on Linux. Mac might be an issue, as we are using OpenGL 4.6 for visualization.
## Requirements
The project is setup using CUDA 12.5. Older architectures and CUDA versions will work too, but you'll have to manually fix the Visual Studio solution to make it work.

We use a number of dependencies which should install automatically via vcpkg in Visual Studio 2022. If manual installation with vcpkg is needed, use the provided
manifest file or install them yourself:
`assimp,entt,stb,glm,glfw3,glad[extensions],imgui[glfw-binding,opengl3-binding],tinyfiledialogs,nlohmann-json`

# Code
The relevant simulation code is in `core/geo`, with CUDA kernels in `core/geo/device`. We further provide the scenes used in our paper in `scenes.zip`, which can be loaded with the application UI.

# Further screenshots
Additional screenshots can be found in `screenshots.zip`.
