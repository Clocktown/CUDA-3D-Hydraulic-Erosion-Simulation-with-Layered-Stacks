#pragma once

#include "../components/terrain.hpp"
#include "performance.hpp"
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
		int renderSand{ true };
		int renderWater{ true };
		bool renderScene{ true };
		bool useRaymarching{ false };
		float surfaceVolumePercentage{ 0.5f };
		float smoothingRadiusInCells{ 1.f };
		float normalSmoothingFactor{ 1.f };
		float aoRadius{ 2.f };
		int missCount{ 8 };
		int fineMissCount{ 16 };
		int debugLayer{ -2 };

		float softShadowScale{ 0.01f };
		float maxSoftShadowDiv{ 0.5f };

		bool accurateNormals{ false };
		bool useExpensiveNormals{ false };
		bool enableWaterAbsorption{ true };
		bool useCheapAbsorption{ false };
		bool enableReflections{ true };
		bool enableRefraction{ true };
		bool enableShadows{ true };
		bool enableSoftShadows{ true };
		bool fixLightLeaks{ false };
		bool enableAO{ true };
		bool enableShadowsInReflection{ true };
		bool enableShadowsInRefraction{ true };
		bool enableAOInReflection{ true };
		bool enableAOInRefraction{ true };
	} rendering;


	std::filesystem::path lastFile{};
	performance performance{};

	bool visable{ true };
private:
	void updatePerformance();
	void updateFile();
	void updateApplication();
	void updateCamera();
	void updateTerrain();
	void updateSimulation();
	void updateRendering();
	void saveToFile(const std::filesystem::path& file, bool jsonOnly = false);
	void loadFromFile(const std::filesystem::path& file);
};

}
