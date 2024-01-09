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
		glm::ivec3 gridSize{ 256, 256, 8 };
		float gridScale{ 1.0f };
	} terrain;
;
	bool visable{ true };
private:
	void updateApplication();
	void updateCamera();
	void updateTerrain();
	void updateSimulation();
	void updateRendering();
};

}