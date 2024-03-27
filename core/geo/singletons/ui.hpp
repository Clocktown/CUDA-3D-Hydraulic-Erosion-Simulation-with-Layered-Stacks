#pragma once

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
	} terrain;

	struct
	{
		glm::vec3 bedrockColor{ 0.5f, 0.5f, 0.5f };
		glm::vec3 sandColor{ 0.9f, 0.8f, 0.6f };
		glm::vec3 waterColor{ 0.0f, 0.3f, 0.75f };
		int useInterpolation{ true };
	} rendering;

	bool visable{ true };
private:
	void updateApplication();
	void updateCamera();
	void updateTerrain();
	void updateSimulation();
	void updateRendering();
};

}
