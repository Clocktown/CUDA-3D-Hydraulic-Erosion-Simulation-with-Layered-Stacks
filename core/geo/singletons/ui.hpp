#pragma once

#include "../components/terrain.hpp"
#include <onec/onec.hpp>

namespace geo
{

struct UI
{
	void update();

	struct
	{
		entt::entity entity{ entt::null };
	} camera;

	struct
	{
		entt::entity entity{ entt::null };
		glm::ivec2 gridSize{ 256 };
		float gridScale{ 1.0f };
		int maxLayerCount{ 8 };
	} terrain;

	struct
	{
		glm::vec3 bedrockColor{ 0.5f, 0.5f, 0.5f };
		glm::vec3 sandColor{ 0.9f, 0.8f, 0.6f };
		glm::vec3 waterColor{ 0.2f, 0.4f, 0.7f };
		int useInterpolation{ true };
	} rendering;

	bool visable{ true };
private:
	void updateFile();
	void updateApplication();
	void updateCamera();
	void updateTerrain();
	void updateSimulation();
	void updateRendering();
	void saveToFile(const std::filesystem::path& file);
	void loadFromFile(const std::filesystem::path& file);
};

}
