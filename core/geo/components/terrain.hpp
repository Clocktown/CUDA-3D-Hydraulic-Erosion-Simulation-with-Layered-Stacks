#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct Settings
{
	float deltaTime{ 1.0f / 60.0f };
	float gravity{ -9.81f };
	float rain{ 0.0f };
	float evaporation{ 0.0f };

	bool init{ true };
	bool paused{ true };
};

struct Terrain
{
	static constexpr int maxLayerCount{ 8 };

	explicit Terrain() = default;
	explicit Terrain(glm::ivec2 gridSize, float gridScale, const Settings& settings = Settings{});

	glm::ivec2 gridSize;
	float gridScale;
	onec::GraphicsBuffer layerCountBuffer;
	onec::GraphicsBuffer heightBuffer;
	Settings settings;
};

}
