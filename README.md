# CUDA 3D Hydraulic Erosion Simulation with layered stacks
This repository contains the full source code of our VMV 2024 paper **3D Real-Time Hydraulic Erosion Simulation using Multi-Layered Heightmaps**

https://github.com/user-attachments/assets/01f3300f-136c-45f1-a7f6-2575fb5c327e

https://github.com/user-attachments/assets/14f66cac-3a0f-4675-9013-a3f5b6d9ccd4

https://github.com/user-attachments/assets/7b07614f-7a43-453e-8c7a-bf9a6bec8a9e

https://github.com/user-attachments/assets/3204772e-0238-4b39-a8ae-8ce4da10def2

https://github.com/user-attachments/assets/0330099b-ac3e-49e1-b2a3-d68ecf776aec

https://github.com/user-attachments/assets/dc8200e6-e799-47f1-ac2f-6c7e08fb93dd

https://github.com/user-attachments/assets/8a7af470-0fca-45e9-8a5a-a483a98b07af

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
