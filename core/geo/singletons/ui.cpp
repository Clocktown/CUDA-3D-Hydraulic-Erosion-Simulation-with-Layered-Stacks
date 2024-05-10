#include "ui.hpp"
#include "imgui.h"
#include "../components/terrain.hpp"
#include "../components/point_renderer.hpp"
#include "../resources/simple_material.hpp"
#include "../device/simulation.hpp"
#include <onec/onec.hpp>
#include <tinyfiledialogs/tinyfiledialogs.h>
#include <tinyexr.h>
#include <nlohmann/json.hpp>
#include <fstream>

namespace geo
{

void UI::update()
{
	const onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	const std::string fps{ std::to_string(application.getFrameRate()) + "fps" };
	const std::string ms{ std::to_string(1000.0 * application.getUnscaledDeltaTime()) + "ms" };
	const std::string title{ application.getName() + " @ " + fps + " / " + ms };
	window.setTitle(title);

	const bool visable{ this->visable != window.isKeyPressed(GLFW_KEY_ESCAPE) };
	this->visable = visable;

	if (!visable)
	{
		return;
	}

	ImGui::Begin("UI", &this->visable);

	updateFile();
	updateApplication();
	updateCamera();
	updateTerrain();
	updateSimulation();
	updateRendering();

	ImGui::End();
}

void UI::updateFile()
{

	if (ImGui::TreeNode("File"))
	{
		if (ImGui::Button("Open"))
		{
			char const* filterPatterns[1] = { "*.json" };
			auto input = tinyfd_openFileDialog("Save JSON", "./scene.json", 1, filterPatterns, "JSON (.json)", 0);

			if (input != nullptr)
			{
				loadFromFile(input);
			}
		}

		if (ImGui::Button("Save"))
		{
			char const* filterPatterns[1] = { "*.json" };
			auto output = tinyfd_saveFileDialog("Save JSON", "./scene.json", 1, filterPatterns, "JSON (.json)");

			if (output != nullptr)
			{
				saveToFile(output);
			}
		}

		ImGui::TreePop();
	}
}

void UI::updateApplication()
{
	onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };

	if (ImGui::TreeNode("Application"))
	{
		int targetFrameRate{ application.getTargetFrameRate() };
		bool vSync{ window.getSwapInterval() != 0 };

		if (ImGui::DragInt("Target Frame Rate", &targetFrameRate, 0.5f, 30, 3000))
		{
			application.setTargetFrameRate(targetFrameRate);
		}

		if (ImGui::Checkbox("Vertical Synchronization", &vSync))
		{
			window.setSwapInterval(vSync);
		}

		ImGui::TreePop();
	}
}

void UI::updateCamera()
{
	if (ImGui::TreeNode("Camera"))
	{
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ camera.entity };

		ONEC_ASSERT(world.hasComponent<onec::PerspectiveCamera>(entity), "Entity must have a perspective camera component");

		onec::PerspectiveCamera& perspectiveCamera{ *world.getComponent<onec::PerspectiveCamera>(entity) };
		float& exposure{ world.getComponent<onec::Exposure>(entity)->exposure };

		float fieldOfView{ glm::degrees(perspectiveCamera.fieldOfView) };

		if (ImGui::DragFloat("Field Of View [deg]", &fieldOfView, 0.1f, 0.367f, 173.0f))
		{
			perspectiveCamera.fieldOfView = glm::radians(fieldOfView);
		}

		ImGui::DragFloat("Near Plane [m]", &perspectiveCamera.nearPlane, 0.01f, 0.001f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Far Plane [m]", &perspectiveCamera.farPlane, 0.5f, 0.001f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Exposure", &exposure, 0.000001f, 0.0f, std::numeric_limits<float>::max(), "%f");

		ImGui::TreePop();
	}
}

void UI::updateTerrain()
{
	if (ImGui::TreeNode("Terrain"))
	{
		if (ImGui::Button("Reset"))
		{
			onec::World& world{ onec::getWorld() };
			const entt::entity entity{ terrain.entity };

			geo::SimpleMaterialUniforms uniforms;
			uniforms.bedrockColor = onec::sRGBToLinear(rendering.bedrockColor);
			uniforms.sandColor = onec::sRGBToLinear(rendering.sandColor);
			uniforms.waterColor = onec::sRGBToLinear(rendering.waterColor);
			uniforms.useInterpolation = rendering.useInterpolation;
			uniforms.gridSize = terrain.gridSize;
			uniforms.gridScale = terrain.gridScale;

			Terrain& terrain{ *world.getComponent<Terrain>(entity) };
			terrain = Terrain{ uniforms.gridSize, uniforms.gridScale, terrain.simulation };
			terrain.simulation.init = true;

			uniforms.maxLayerCount = terrain.maxLayerCount;
			uniforms.layerCounts = terrain.layerCountBuffer.getBindlessHandle();
			uniforms.heights = terrain.heightBuffer.getBindlessHandle();
			uniforms.stability = terrain.stabilityBuffer.getBindlessHandle();
			uniforms.indices = terrain.indicesBuffer.getBindlessHandle();

			world.setComponent<onec::Position>(entity, -0.5f * uniforms.gridScale * world.getComponent<onec::Scale>(entity)->scale * glm::vec3{ uniforms.gridSize.x, 0.0f, uniforms.gridSize.y });

			PointRenderer& pointRenderer{ *world.getComponent<PointRenderer>(entity) };
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&uniforms, 1));
			pointRenderer.count = terrain.numValidColumns;
		}

		ImGui::DragInt2("Grid Size", &terrain.gridSize.x, 0.5f, 16, 4096);
		ImGui::DragFloat("Grid Scale [m]", &terrain.gridScale, 0.01f, 0.001f, 10.0f);

		ImGui::TreePop();
	}
}

void UI::updateSimulation()
{
	if (ImGui::TreeNode("Simulation"))
	{
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ terrain.entity };

		Simulation& simulation{ world.getComponent<Terrain>(entity)->simulation };

		if (simulation.paused)
		{
			if (ImGui::Button("Start"))
			{
				simulation.paused = false;
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
			{
				simulation.paused = true;
			}
		}

		ImGui::DragFloat("Delta Time [s]", &simulation.deltaTime, 0.01f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Gravity [m/s^2]", &simulation.gravity, 0.1f);
		ImGui::DragFloat("Rain [m/(m^2 * s)]", &simulation.rain, 0.01f, 0.0f, std::numeric_limits<float>::max());

		float evaporation{ 100.0f * simulation.evaporation };

		if (ImGui::DragFloat("Evaporation [%/s]", &evaporation, 0.5f, 0.0f, std::numeric_limits<float>::max()))
		{
			simulation.evaporation = 0.01f * evaporation;
		}

		float petrification{ 100.0f * simulation.petrification };

		if (ImGui::DragFloat("Petrification [%/s]", &petrification, 0.01f, 0.0f, std::numeric_limits<float>::max()))
		{
			simulation.petrification = 0.01f * petrification;
		}

		ImGui::DragFloat("Sediment Capacity Constant", &simulation.sedimentCapacityConstant, 0.01f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Bedrock Dissolving Constant", &simulation.bedrockDissolvingConstant, 0.1f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Sand Dissolving Constant", &simulation.sandDissolvingConstant, 0.1f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Sediment Deposition Constant", &simulation.sedimentDepositionConstant, 0.1f, 0.0f, std::numeric_limits<float>::max());

		float minTerrainAngle{ glm::degrees(simulation.minTerrainAngle) };

		if (ImGui::DragFloat("Min. Terrain Angle [deg]", &minTerrainAngle, 0.1f, 0.0f, 90.0f))
		{
			simulation.minTerrainAngle = glm::radians(minTerrainAngle);
		}

		float dryTalusAngle{ glm::degrees(simulation.dryTalusAngle) };

		if (ImGui::DragFloat("Dry Talus Angle [deg]", &dryTalusAngle, 0.1f, 0.0f, 90.0f))
		{
			simulation.dryTalusAngle = glm::radians(dryTalusAngle);
		}

		float wetTalusAngle{ glm::degrees(simulation.wetTalusAngle) };

		if (ImGui::DragFloat("Wet Talus Angle [deg]", &wetTalusAngle, 0.1f, 0.0f, 90.0f))
		{
			simulation.wetTalusAngle = glm::radians(wetTalusAngle);
		}

		ImGui::DragFloat("Min. Horizontal Erosion", &simulation.minHorizontalErosion, 0.001f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Horizontal Erosion Strength", &simulation.horizontalErosionStrength, 0.001f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Min. Split Damage", &simulation.minSplitDamage, 0.1f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Split Threshold", &simulation.splitThreshold, 0.01f, 0.0f, 1.0f);

		if (ImGui::TreeNode("Support Check")) {

			ImGui::DragFloat("Bedrock density", &simulation.bedrockDensity);
			ImGui::DragFloat("Sand density", &simulation.sandDensity);
			ImGui::DragFloat("Bedrock support", &simulation.bedrockSupport);
			ImGui::DragFloat("Border support", &simulation.borderSupport);
			ImGui::DragInt("Max. stability propagation steps", &simulation.maxStabilityPropagationSteps);
			ImGui::DragInt("Stability propagation steps per iteration", &simulation.stabilityPropagationStepsPerIteration);

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}

void UI::updateRendering()
{
	if (ImGui::TreeNode("Rendering"))
	{
		if (ImGui::Button("Reload Shaders")) {
			const std::filesystem::path assets{ onec::getApplication().getDirectory() / "assets" };
			const std::array<std::filesystem::path, 3> shaders{ assets / "shaders/point_terrain/simple_terrain.vert",
																assets / "shaders/point_terrain/simple_terrain.geom",
																assets / "shaders/point_terrain/simple_terrain.frag" };

			PointRenderer& pointRenderer{ *onec::getWorld().getComponent<PointRenderer>(terrain.entity)};
			pointRenderer.material->program = std::make_shared<onec::Program>(shaders);
		}
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ terrain.entity };

		Terrain& terrain{ *world.getComponent<Terrain>(entity) };
		float& scale{ world.getComponent<onec::Scale>(entity)->scale };
		PointRenderer& pointRenderer{ *world.getComponent<PointRenderer>(entity) };
		
		if (ImGui::DragFloat("Visual Scale", &scale, 0.1f, 0.001f, std::numeric_limits<float>::max()))
		{
			world.setComponent<onec::Position>(entity, -0.5f * terrain.gridScale * scale * glm::vec3{ terrain.gridSize.x, 0.0f, terrain.gridSize.y });
		}

		ImGui::DragFloat("Ambient Light [lux]", &world.getSingleton<onec::AmbientLight>()->strength, 1.0f, 0.0f, std::numeric_limits<float>::max());
		ImGui::ColorEdit3("Background Color", &world.getSingleton<onec::AmbientLight>()->color.x);

		if (ImGui::ColorEdit3("Bedrock Color", &rendering.bedrockColor.x))
		{
			const glm::vec3 bedrockColor{ onec::sRGBToLinear(rendering.bedrockColor) };
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&bedrockColor, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, bedrockColor)), sizeof(glm::vec3));
		}

		if (ImGui::ColorEdit3("Sand Color", &rendering.sandColor.x))
		{
			const glm::vec3 sandColor{ onec::sRGBToLinear(rendering.sandColor) };
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&sandColor, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, sandColor)), sizeof(glm::vec3));
		}

		if (ImGui::ColorEdit3("Water Color", &rendering.waterColor.x))
		{
			const glm::vec3 waterColor{ onec::sRGBToLinear(rendering.waterColor) };
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&waterColor, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, waterColor)), sizeof(glm::vec3));
		}

		bool useInterpolation = bool(rendering.useInterpolation);
		if (ImGui::Checkbox("Use Interpolation", &useInterpolation)) {
			rendering.useInterpolation = int(useInterpolation);
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&rendering.useInterpolation, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, useInterpolation)), sizeof(int));
		}

		ImGui::TreePop();
	}
}

void UI::saveToFile(const std::filesystem::path& file)
{
	const onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	onec::PerspectiveCamera& camera{ *world.getComponent<onec::PerspectiveCamera>(this->camera.entity) };
	Terrain& terrain{ *world.getComponent<Terrain>(this->terrain.entity) };
	Simulation& simulation{ terrain.simulation };

	nlohmann::json json;
	
	json["Application/TargetFrameRate"] = application.getTargetFrameRate();
	json["Application/VerticalSynchronization"] = window.getSwapInterval() != 0;

	const glm::vec3 cameraPosition{ world.getComponent<onec::Position>(this->camera.entity)->position };
	const glm::quat cameraRotation{ world.getComponent<onec::Rotation>(this->camera.entity)->rotation };
	json["Camera/Position"] = { cameraPosition.x, cameraPosition.y, cameraPosition.z };
	json["Camera/Rotation"] = { cameraRotation.x, cameraRotation.y, cameraRotation.z, cameraRotation.w };
	json["Camera/FieldOfView"] = camera.fieldOfView;
	json["Camera/NearPlane"] = camera.nearPlane;
	json["Camera/FarPlane"] = camera.farPlane;
	json["Camera/Exposure"] = world.getComponent<onec::Exposure>(this->camera.entity)->exposure;

	json["Terrain/GridSize"] = { terrain.gridSize.x, terrain.gridSize.y };
	json["Terrain/GridScale"] = terrain.gridScale;

	json["Simulation/DeltaTime"] = simulation.deltaTime;
	json["Simulation/Gravity"] = simulation.gravity;
	json["Simulation/Rain"] = simulation.rain;
	json["Simulation/Evaporation"] = simulation.evaporation;
	json["Simulation/Petrification"] = simulation.petrification;
	json["Simulation/SedimentCapacityConstant"] = simulation.sedimentCapacityConstant;
	json["Simulation/BedrockDissolvingConstant"] = simulation.bedrockDissolvingConstant;
	json["Simulation/SandDissolvingConstant"] = simulation.sandDissolvingConstant;
	json["Simulation/SedimentDepositionConstant"] = simulation.sedimentDepositionConstant;
	json["Simulation/MinTerrainAngle"] = simulation.minTerrainAngle;
	json["Simulation/DryTalusAngle"] = simulation.dryTalusAngle;
	json["Simulation/WetTalusAngle"] = simulation.wetTalusAngle;
	json["Simulation/MinHorizontalErosion"] = simulation.minHorizontalErosion;
	json["Simulation/HorizontalErosionStrength"] = simulation.horizontalErosionStrength;
	json["Simulation/MinSplitDamage"] = simulation.minSplitDamage;
	json["Simulation/SplitThreshold"] = simulation.splitThreshold;

	const auto backgroundColor{ world.getSingleton<onec::AmbientLight>()->color };
	json["Rendering/VisualScale"] = world.getComponent<onec::Scale>(this->terrain.entity)->scale;
	json["Rendering/AmbientLight"] = world.getSingleton<onec::AmbientLight>()->strength;
	json["Rendering/BackgroundColor"] = { backgroundColor.r, backgroundColor.g, backgroundColor.b };
	json["Rendering/BedrockColor"] = { rendering.bedrockColor.r, rendering.bedrockColor.g, rendering.bedrockColor.b };
	json["Rendering/SandColor"] = { rendering.sandColor.r, rendering.sandColor.g, rendering.sandColor.b };
	json["Rendering/WaterColor"] = { rendering.waterColor.r, rendering.waterColor.g, rendering.waterColor.b };
	json["Rendering/UseInterpolation"] = rendering.useInterpolation;

	std::ptrdiff_t byteCount{ 0 };
	byteCount += terrain.layerCountBuffer.getCount();
	byteCount += terrain.heightBuffer.getCount();
	byteCount += terrain.fluxBuffer.getCount();
	byteCount += terrain.stabilityBuffer.getCount();
	byteCount += terrain.sedimentBuffer.getCount();
	byteCount += terrain.damageBuffer.getCount();

	std::vector<std::byte> destination(byteCount);
	std::ptrdiff_t i{ 0 };

	terrain.layerCountBuffer.download({ destination.data() + i, terrain.layerCountBuffer.getCount() }); i += terrain.layerCountBuffer.getCount();
	terrain.heightBuffer.download({ destination.data() + i, terrain.heightBuffer.getCount() }); i += terrain.heightBuffer.getCount();
	terrain.fluxBuffer.download({ destination.data() + i, terrain.fluxBuffer.getCount() }); i += terrain.fluxBuffer.getCount();
	terrain.stabilityBuffer.download({ destination.data() + i, terrain.stabilityBuffer.getCount() }); i += terrain.stabilityBuffer.getCount();
	terrain.sedimentBuffer.download({ destination.data() + i, terrain.sedimentBuffer.getCount() }); i += terrain.sedimentBuffer.getCount();
	terrain.damageBuffer.download({ destination.data() + i, terrain.damageBuffer.getCount() }); i += terrain.damageBuffer.getCount();

	onec::writeFile(file, json.dump());
	onec::writeFile(file.stem().concat(".dat"), std::string_view{ reinterpret_cast<char*>(destination.data()), static_cast<std::size_t>(byteCount) });
}

void UI::loadFromFile(const std::filesystem::path& file)
{
	onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	onec::PerspectiveCamera& camera{ *world.getComponent<onec::PerspectiveCamera>(this->camera.entity) };
	Terrain& terrain{ *world.getComponent<Terrain>(this->terrain.entity) };

	nlohmann::json json{ nlohmann::json::parse(std::ifstream{ file }) };

	application.setTargetFrameRate(json["Application/TargetFrameRate"]);
	window.setSwapInterval(json["Application/VerticalSynchronization"]);

	world.getComponent<onec::Position>(this->camera.entity)->position = glm::vec3{ json["Camera/Position"][0], json["Camera/Position"][1], json["Camera/Position"][2] };
	world.getComponent<onec::Rotation>(this->camera.entity)->rotation = glm::quat{ json["Camera/Rotation"][3], json["Camera/Rotation"][0], json["Camera/Rotation"][1], json["Camera/Rotation"][2] };
	camera.fieldOfView = json["Camera/FieldOfView"];
	camera.nearPlane = json["Camera/NearPlane"];
	camera.farPlane = json["Camera/FarPlane"];
	world.getComponent<onec::Exposure>(this->camera.entity)->exposure = json["Camera/Exposure"];
	
	this->terrain.gridSize = glm::ivec2{ json["Terrain/GridSize"][0], json["Terrain/GridSize"][1] };
	this->terrain.gridScale = json["Terrain/GridScale"];

	terrain = Terrain{ this->terrain.gridSize, this->terrain.gridScale };
	const std::string source{ onec::readFile(file.stem().concat(".dat")) };
	const std::byte* const data{ reinterpret_cast<const std::byte*>(source.data()) };

	std::ptrdiff_t i{ 0 };
	terrain.layerCountBuffer.upload({ data + i, terrain.layerCountBuffer.getCount() }); i += terrain.layerCountBuffer.getCount();
	terrain.heightBuffer.upload({ data + i, terrain.heightBuffer.getCount() }); i += terrain.heightBuffer.getCount();
	terrain.fluxBuffer.upload({ data + i, terrain.fluxBuffer.getCount() }); i += terrain.fluxBuffer.getCount();
	terrain.stabilityBuffer.upload({ data + i, terrain.stabilityBuffer.getCount() }); i += terrain.stabilityBuffer.getCount();
	terrain.sedimentBuffer.upload({ data + i, terrain.sedimentBuffer.getCount() }); i += terrain.sedimentBuffer.getCount();
	terrain.damageBuffer.upload({ data + i, terrain.damageBuffer.getCount() }); i += terrain.damageBuffer.getCount();

	Simulation& simulation{ terrain.simulation };
	simulation.deltaTime = json["Simulation/DeltaTime"];
	simulation.gravity = json["Simulation/Gravity"];
	simulation.rain = json["Simulation/Rain"];
	simulation.evaporation = json["Simulation/Evaporation"];
	simulation.petrification = json["Simulation/Petrification"];
	simulation.sedimentCapacityConstant = json["Simulation/SedimentCapacityConstant"];
	simulation.bedrockDissolvingConstant = json["Simulation/BedrockDissolvingConstant"];
	simulation.sandDissolvingConstant = json["Simulation/SandDissolvingConstant"];
	simulation.sedimentDepositionConstant = json["Simulation/SedimentDepositionConstant"];
	simulation.minTerrainAngle = json["Simulation/MinTerrainAngle"];
	simulation.dryTalusAngle = json["Simulation/DryTalusAngle"];
	simulation.wetTalusAngle = json["Simulation/WetTalusAngle"];
	simulation.minHorizontalErosion = json["Simulation/MinHorizontalErosion"];
	simulation.horizontalErosionStrength = json["Simulation/HorizontalErosionStrength"];
	simulation.minSplitDamage = json["Simulation/MinSplitDamage"];
	simulation.splitThreshold = json["Simulation/SplitThreshold"];
	simulation.init = false;
	simulation.paused = true;

	world.getComponent<onec::Scale>(this->terrain.entity)->scale = json["Rendering/VisualScale"];
	world.getSingleton<onec::AmbientLight>()->color = glm::vec3{ json["Rendering/BackgroundColor"][0], json["Rendering/BackgroundColor"][1], json["Rendering/BackgroundColor"][2] };
	world.getSingleton<onec::AmbientLight>()->strength = json["Rendering/AmbientLight"];
	rendering.bedrockColor = glm::vec3{ json["Rendering/BedrockColor"][0], json["Rendering/BedrockColor"][1], json["Rendering/BedrockColor"][2] };
	rendering.sandColor = glm::vec3{ json["Rendering/SandColor"][0], json["Rendering/SandColor"][1], json["Rendering/SandColor"][2] };
	rendering.waterColor = glm::vec3{ json["Rendering/WaterColor"][0], json["Rendering/WaterColor"][1], json["Rendering/WaterColor"][2] };
	rendering.useInterpolation = json["Rendering/UseInterpolation"];

    PointRenderer& pointRenderer{ *onec::getWorld().getComponent<PointRenderer>(this->terrain.entity) };
	geo::SimpleMaterialUniforms uniforms;
	uniforms.bedrockColor = onec::sRGBToLinear(rendering.bedrockColor);
	uniforms.sandColor = onec::sRGBToLinear(rendering.sandColor);
	uniforms.waterColor = onec::sRGBToLinear(rendering.waterColor);
	uniforms.useInterpolation = rendering.useInterpolation;
	uniforms.gridSize = terrain.gridSize;
	uniforms.gridScale = terrain.gridScale;
	uniforms.maxLayerCount = terrain.maxLayerCount;
	uniforms.layerCounts = terrain.layerCountBuffer.getBindlessHandle();
	uniforms.heights = terrain.heightBuffer.getBindlessHandle();
	uniforms.stability = terrain.stabilityBuffer.getBindlessHandle();
	uniforms.indices = terrain.indicesBuffer.getBindlessHandle();
	pointRenderer.material->uniformBuffer.initialize(onec::asBytes(&uniforms, 1));
}

}
