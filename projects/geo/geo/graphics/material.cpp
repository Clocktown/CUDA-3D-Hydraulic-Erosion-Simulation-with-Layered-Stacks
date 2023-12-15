#include "material.hpp"

namespace geo
{

Material::Material(Terrain& terrain)
{
	const onec::Application& application{ onec::getApplication() };

	uniforms.gridSize = terrain.gridSize;
	uniforms.gridScale = terrain.gridScale;
	uniforms.infoMap = terrain.infoMap.getBindlessHandle();
	uniforms.heightMap = terrain.heightMap.getBindlessHandle();
	uniformBuffer.initialize(onec::asBytes(&uniforms, 1));

	const std::filesystem::path assets{ application.getDirectory() / "assets" };
	const std::array<std::filesystem::path, 3> files{ assets / "shaders/terrain/terrain.vert",
											          assets / "shaders/terrain/terrain.geom",
											          assets / "shaders/terrain/terrain.frag" };

	program = std::make_shared<onec::Program>(files);
	renderState = std::make_shared<onec::RenderState>();
}

}
