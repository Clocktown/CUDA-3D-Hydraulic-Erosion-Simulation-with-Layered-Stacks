#pragma once

#include "../graphics/terrain.hpp"
#include "../graphics/terrain_brdf.hpp"
#include <onec/onec.hpp>

namespace geo
{

struct Editor
{
	struct Application
	{
		int targetFramerate{ 3000 };
		float timeScale{ 1.0f };
		float fixedDeltaTime{ 1.0f / 60.0f };
		bool isVSyncEnabled{ false };
	};

	struct Camera
	{
		entt::entity entity{ entt::null };
		float fieldOfView{ 60.0f };
		float nearPlane{ 0.1f };
		float farPlane{ 1000.0f };
	};

	struct Terrain
	{
		entt::entity entity{ entt::null };
		std::shared_ptr<geo::Terrain> terrain{ nullptr };
		glm::ivec2 gridSize{ 256 };
		float gridScale{ 1.0f };
		int maxLayerCount{ 8 };
	};

	struct Simulation
	{
		struct Rain
		{
			float amount{ 500.0f };
			bool isPaused{ true };
		};

		Rain rain;
		bool isPaused{ true };
	};

	struct Rendering
	{
		glm::vec3 backgroundColor{ 0.7f, 0.9f, 1.0f };
		glm::vec3 bedrockColor{ 0.5f, 0.5f, 0.5f };
		glm::vec3 sandColor{ 0.9f, 0.8f, 0.6f };
		glm::vec3 waterColor{ 0.1f, 0.1f, 1.0f };
		std::shared_ptr<TerrainBRDF> material{ nullptr };
	};

	Application application;
	Camera camera;
	Terrain terrain;
	Simulation simulation;
	Rendering rendering;
};

}