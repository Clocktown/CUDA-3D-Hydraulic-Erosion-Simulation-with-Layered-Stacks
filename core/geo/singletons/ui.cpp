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
#include <zlib.h>
#include <fstream>

namespace geo
{

void UI::update()
{
	const onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	//const std::string fps{ std::to_string(application.getFrameRate()) + "fps" };
	//const std::string ms{ std::to_string(1000.0 * application.getUnscaledDeltaTime()) + "ms" };
	//const std::string title{ application.getName() + " @ " + fps + " / " + ms };
	//window.setTitle(title);

	const bool visable{ this->visable != window.isKeyPressed(GLFW_KEY_ESCAPE) };
	this->visable = visable;

	if (!visable)
	{
		return;
	}

	ImGui::Begin("UI", &this->visable);

	updatePerformance();
	updateFile();
	updateApplication();
	updateCamera();
	updateTerrain();
	updateSimulation();
	updateRendering();

	ImGui::End();
}

void UI::updatePerformance() {
	ImGui::LabelText("Frametime", "%f [ms]", performance.measurements["Frametime"].mean);

	if (ImGui::TreeNode("Performance")) {
		onec::World& world{onec::getWorld()};
		const entt::entity entity{ terrain.entity };
		Simulation& simulation{ world.getComponent<Terrain>(entity)->simulation };

		if (ImGui::Button("Reset")) {
			performance.resetAll();
		}
		ImGui::LabelText("Step", "%i", simulation.currentSimulationStep);
		ImGui::DragInt("Pause every n steps", &performance.pauseAfterStepCount, 0.1f, 0, 100000);
		ImGui::Checkbox("Measure Performance", &performance.measurePerformance);
		ImGui::Checkbox("Measure Rendering", &performance.measureRendering);
		ImGui::Checkbox("Measure Categories", &performance.measureParts);
		ImGui::Checkbox("Measure All Kernels", &performance.measureIndividualKernels);
		float rain = performance.measurements["Rain"].mean;

		float setupPipes = performance.measurements["Setup Pipes"].mean;
		float resolvePipes = performance.measurements["Resolve Pipes"].mean;
		float transport = performance.measureIndividualKernels ? setupPipes + resolvePipes : performance.measurements["Transport"].mean;

		float horizontalErosion = performance.measurements["Horizontal Erosion"].mean;
		float splitKernel = performance.measurements["Split Kernel"].mean;
		float verticalErosion = performance.measurements["Vertical Erosion"].mean;
		float erosion = performance.measureIndividualKernels ? horizontalErosion + splitKernel + verticalErosion : performance.measurements["Erosion"].mean;

		float endSupport = performance.measurements["End Support Check"].mean;
		float stepSupport = performance.measurements["Step Support Check"].mean;
		float startSupport = performance.measurements["Start Support Check"].mean;
		float support = performance.measureIndividualKernels ? (float(simulation.stabilityPropagationStepsPerIteration) / simulation.maxStabilityPropagationSteps) * (startSupport + endSupport) + stepSupport : performance.measurements["Support"].mean;

		float totalSim = (performance.measureParts || performance.measureIndividualKernels) ? rain + transport + erosion + support : performance.measurements["Global Simulation"].mean;

		ImGui::LabelText("Total Simulation", "%f [ms]", totalSim);
		if (ImGui::TreeNode("Simulation Details")) {
			ImGui::LabelText("Rain", "%f [ms]", rain);
			ImGui::LabelText("Transport", "%f [ms]", transport);
			if (ImGui::TreeNode("Transport Details")) {
				ImGui::LabelText("Setup Pipes", "%f [ms]", setupPipes);
				ImGui::LabelText("Resolve Pipes", "%f [ms]", resolvePipes);
			
				ImGui::TreePop();
			}

			ImGui::LabelText("Erosion", "%f [ms]", erosion);
			if (ImGui::TreeNode("Erosion Details")) {
				ImGui::LabelText("Horizontal Erosion", "%f [ms]", horizontalErosion);
				ImGui::LabelText("Split Kernel", "%f [ms]", splitKernel);
				ImGui::LabelText("Vertical Erosion", "%f [ms]", verticalErosion);
			
				ImGui::TreePop();
			}

			ImGui::LabelText("Support", "%f [ms]", support);
			if (ImGui::TreeNode("Support Details")) {
				ImGui::LabelText("Start Support Check", "%f [ms]", startSupport);
				ImGui::LabelText("Step Support Check", "%f [ms]", stepSupport / simulation.stabilityPropagationStepsPerIteration);
				ImGui::LabelText("End Support Check", "%f [ms]", endSupport);
			
				ImGui::TreePop();
			}

			ImGui::TreePop();
		}

		ImGui::LabelText("Total Rendering", "%f [ms]", performance.measurements["Rendering"].mean + performance.measurements["Build Draw List"].mean);
		if (ImGui::TreeNode("Rendering Details")) {
			ImGui::LabelText("Draw Call", "%f [ms]", performance.measurements["Rendering"].mean);
			ImGui::LabelText("Build Draw List", "%f [ms]", performance.measurements["Build Draw List"].mean);

			ImGui::TreePop();
		}
		
		ImGui::TreePop();
	}
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
			uniforms.renderSand = rendering.renderSand;
			uniforms.renderWater = rendering.renderWater;
			uniforms.gridSize = terrain.gridSize;
			uniforms.gridScale = terrain.gridScale;
			uniforms.maxLayerCount = terrain.maxLayerCount;

			Terrain& terrain{ *world.getComponent<Terrain>(entity) };
			terrain = Terrain{ uniforms.gridSize, uniforms.gridScale, static_cast<char>(uniforms.maxLayerCount), terrain.simulation };
			terrain.simulation.init = true;

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
		ImGui::DragInt("Max. Layer Count", &terrain.maxLayerCount, 0.01f, 1, 127);

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
		
		if (ImGui::TreeNode("General")) {
			
			ImGui::Checkbox("Use Outflow Borders", &simulation.useOutflowBorders);
			ImGui::DragFloat("Delta Time [s]", &simulation.deltaTime, 0.01f, 0.0f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Gravity [m/s^2]", &simulation.gravity, 0.1f);
			ImGui::DragFloat("Rain [m/(m^2 * s)]", &simulation.rain, 0.01f, 0.0f, std::numeric_limits<float>::max());

			float evaporation{ 100.0f * simulation.evaporation };

			if (ImGui::DragFloat("Evaporation [%/s]", &evaporation, 0.5f, 0.0f, std::numeric_limits<float>::max()))
			{
				simulation.evaporation = 0.01f * evaporation;
			}
			ImGui::DragFloat("Evaporation ceiling falloff", &simulation.evaporationEmptySpaceScale, 0.01f, 0.f, 100.f);

			float petrification{ 100.0f * simulation.petrification };

			if (ImGui::DragFloat("Petrification [%/s]", &petrification, 0.01f, 0.0f, std::numeric_limits<float>::max()))
			{
				simulation.petrification = 0.01f * petrification;
			}

			for (int i = 0; i < 4; ++i) {
				std::string label = "Source ";
				label += std::to_string(i + 1);
				if (ImGui::TreeNode(label.c_str())) {
					ImGui::DragFloat("Strength", &simulation.sourceStrengths[i], 0.01f, 0.0f, std::numeric_limits<float>::max());
					ImGui::DragFloat("Radius", &simulation.sourceSize[i], 0.01f, 0.0f, 1000.f);
					ImGui::DragInt2("Location", &simulation.sourceLocations[i].x, 0.1f, 0, glm::max(terrain.gridSize.x, terrain.gridSize.y));

					ImGui::TreePop();
				}
			}

			ImGui::TreePop();
		}

		ImGui::Checkbox("##enableSlippage", &simulation.slippageEnabled);
		ImGui::SameLine();
		if (ImGui::TreeNode("Sand Slippage")) {

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

			ImGui::DragFloat("Interpolation range", &simulation.slippageInterpolationRange, 0.01f, 0.f, 10.f);

			ImGui::TreePop();
		}

		ImGui::Checkbox("##enableVerticalErosion", &simulation.verticalErosionEnabled);
		ImGui::SameLine();
		if (ImGui::TreeNode("Vertical Erosion")) {

			ImGui::DragFloat("Sediment Capacity Constant", &simulation.sedimentCapacityConstant, 0.01f, 0.0f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Bedrock Dissolving Constant", &simulation.bedrockDissolvingConstant, 0.1f, 0.0f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Sand Dissolving Constant", &simulation.sandDissolvingConstant, 0.1f, 0.0f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Sediment Deposition Constant", &simulation.sedimentDepositionConstant, 0.1f, 0.0f, std::numeric_limits<float>::max());

			float minSlopeErosionScale{ simulation.minSlopeErosionScale };

			if (ImGui::DragFloat("Min. Slope Erosion Scale", &minSlopeErosionScale, 0.001f, 0.0f, 1.0f))
			{
				simulation.minSlopeErosionScale = minSlopeErosionScale;
			}

			float maxSlopeErosionScale{ simulation.maxSlopeErosionScale };

			if (ImGui::DragFloat("Max. Slope Erosion Scale", &maxSlopeErosionScale, 0.001f, minSlopeErosionScale, 10.0f))
			{
				simulation.maxSlopeErosionScale = maxSlopeErosionScale;
			}

			ImGui::DragFloat("Max Erosion Depth", &simulation.erosionWaterMaxHeight, 0.1f, 0.0f, 100.f);
			ImGui::DragFloat("Max Top Erosion Distance", &simulation.topErosionWaterScale, 0.01f, 0.0f, 100.f);
			ImGui::DragFloat("Bedrock Erosion Sand Threshold", &simulation.sandThreshold, 0.0001f, 0.f, 1.f);
			ImGui::DragFloat("Slope Falloff Start", &simulation.verticalErosionSlopeFadeStart, 0.001f, 0.f, 1.f);

			ImGui::TreePop();
		}

		ImGui::Checkbox("##enableHorizontalErosion", &simulation.horizontalErosionEnabled);
		ImGui::SameLine();
		if (ImGui::TreeNode("Horizontal Erosion")) {

			ImGui::DragFloat("Min. Horizontal Erosion Slope (sine)", &simulation.minHorizontalErosionSlope, 0.001f, 0.0f, 1.f);
			ImGui::DragFloat("Horizontal Erosion Strength", &simulation.horizontalErosionStrength, 0.001f, 0.0f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Min. Split Damage", &simulation.minSplitDamage, 0.1f, 0.0f, std::numeric_limits<float>::max());
			ImGui::DragFloat("Split Size", &simulation.splitSize, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("Damage Recovery Rate", &simulation.damageRecovery, 0.001f, 0.0f, 1.0f);

			ImGui::TreePop();
		}

		if (ImGui::Checkbox("##enableSupportCheck", &simulation.supportCheckEnabled)) {
			simulation.stabilityWasReenabled = simulation.supportCheckEnabled;
		}
		ImGui::SameLine();
		if (ImGui::TreeNode("Support Check")) {

			ImGui::Checkbox("Use weight", &simulation.useWeightInSupportCheck);
			ImGui::DragFloat("Bedrock density", &simulation.bedrockDensity);
			ImGui::DragFloat("Sand density", &simulation.sandDensity);
			ImGui::DragFloat("Bedrock support", &simulation.bedrockSupport);
			ImGui::DragFloat("Border support", &simulation.borderSupport);
			ImGui::DragFloat("Min. Bedrock Thickness", &simulation.minBedrockThickness);
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

		bool renderSand = bool(rendering.renderSand);
		if (ImGui::Checkbox("Render Sand", &renderSand)) {
			rendering.renderSand = int(renderSand);
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&rendering.renderSand, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, renderSand)), sizeof(int));
		}

		bool renderWater = bool(rendering.renderWater);
		if (ImGui::Checkbox("Render Water", &renderWater)) {
			rendering.renderWater = int(renderWater);
			pointRenderer.material->uniformBuffer.upload(onec::asBytes(&rendering.renderWater, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, renderWater)), sizeof(int));
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

	onec::Buffer layerCountBuffer{ terrain.layerCountBuffer };
	const int usedLayerCount{ device::getMaxLayerCount(reinterpret_cast<char*>(layerCountBuffer.getData()), layerCountBuffer.getCount()) };
	const std::ptrdiff_t cellCount{ terrain.gridSize.x * terrain.gridSize.y };
	const std::ptrdiff_t columnCount{ cellCount * usedLayerCount };
	const std::ptrdiff_t layerCountBytes{ cellCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::layerCounts)) };
	const std::ptrdiff_t heightBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::heights)) };
	const std::ptrdiff_t fluxBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::fluxes)) };
	const std::ptrdiff_t stabilityBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::stability)) };
	const std::ptrdiff_t sedimentBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::sediments)) };
	const std::ptrdiff_t damageBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::damages)) };
	layerCountBuffer.release();

	uLongf bytes{ 0 };
	bytes += layerCountBytes;
	bytes += heightBytes;
	bytes += fluxBytes;
	bytes += stabilityBytes;
	bytes += sedimentBytes;
	bytes += damageBytes;

	std::vector<std::byte> uncompressed(bytes);
	std::ptrdiff_t i{ 0 };

	terrain.layerCountBuffer.download({ uncompressed.data() + i, layerCountBytes }); i += layerCountBytes;
	terrain.heightBuffer.download({ uncompressed.data() + i, heightBytes }); i += heightBytes;
	terrain.fluxBuffer.download({ uncompressed.data() + i, fluxBytes }); i += fluxBytes;
	terrain.stabilityBuffer.download({ uncompressed.data() + i, stabilityBytes }); i += stabilityBytes;
	terrain.sedimentBuffer.download({ uncompressed.data() + i, sedimentBytes }); i += sedimentBytes;
	terrain.damageBuffer.download({ uncompressed.data() + i, damageBytes }); i += damageBytes;

	uLongf compressedSize{ compressBound(bytes) };
	std::vector<std::byte> compressed(compressedSize);
	const int status{ compress(reinterpret_cast<Bytef*>(compressed.data()), &compressedSize, reinterpret_cast<Bytef*>(uncompressed.data()), uncompressed.size()) };

	ONEC_ASSERT(status == Z_OK, "Failed to compress data");

	json["Terrain/GridSize"] = { terrain.gridSize.x, terrain.gridSize.y };
	json["Terrain/GridScale"] = terrain.gridScale;
	json["Terrain/MaxLayerCount"] = terrain.maxLayerCount;
	json["Terrain/UsedLayerCount"] = usedLayerCount;

	json["Simulation/UseOutflowBorders"] = simulation.useOutflowBorders;
	json["Simulation/DeltaTime"] = simulation.deltaTime;
	json["Simulation/Gravity"] = simulation.gravity;
	json["Simulation/Rain"] = simulation.rain;
	json["Simulation/Evaporation"] = simulation.evaporation;
	json["Simulation/Petrification"] = simulation.petrification;
	json["Simulation/SedimentCapacityConstant"] = simulation.sedimentCapacityConstant;
	json["Simulation/BedrockDissolvingConstant"] = simulation.bedrockDissolvingConstant;
	json["Simulation/SandDissolvingConstant"] = simulation.sandDissolvingConstant;
	json["Simulation/SedimentDepositionConstant"] = simulation.sedimentDepositionConstant;
	json["Simulation/MinSlopeErosionScale"] = simulation.minSlopeErosionScale;
	json["Simulation/MaxSlopeErosionScale"] = simulation.maxSlopeErosionScale;
	json["Simulation/DryTalusAngle"] = simulation.dryTalusAngle;
	json["Simulation/WetTalusAngle"] = simulation.wetTalusAngle;
	json["Simulation/SlippageInterpolationRange"] = simulation.slippageInterpolationRange;
	json["Simulation/MinHorizontalErosionSlope"] = simulation.minHorizontalErosionSlope;
	json["Simulation/VerticalErosionSlopeFadeStart"] = simulation.verticalErosionSlopeFadeStart;
	json["Simulation/HorizontalErosionStrength"] = simulation.horizontalErosionStrength;
	json["Simulation/MinSplitDamage"] = simulation.minSplitDamage;
	json["Simulation/SplitSize"] = simulation.splitSize;
	json["Simulation/DamageRecovery"] = simulation.damageRecovery;
	json["Simulation/EnableVerticalErosion"] = simulation.verticalErosionEnabled;
	json["Simulation/EnableHorizontalErosion"] = simulation.horizontalErosionEnabled;
	json["Simulation/EnableSlippage"] = simulation.slippageEnabled;
	json["Simulation/EnableSupportCheck"] = simulation.supportCheckEnabled;
	json["Simulation/UseWeightInSupportCheck"] = simulation.useWeightInSupportCheck;
	json["Simulation/CurrentSimulationStep"] = simulation.currentSimulationStep;
	json["Simulation/CurrentStabilityStep"] = simulation.currentStabilityStep;
	json["Simulation/MinBedrockThickness"] = simulation.minBedrockThickness;
	json["Simulation/ErosionWaterMaxHeight"] = simulation.erosionWaterMaxHeight;
	json["Simulation/TopErosionWaterScale"] = simulation.topErosionWaterScale;
	json["Simulation/EvaporationEmptySpaceScale"] = simulation.evaporationEmptySpaceScale;
	json["Simulation/SandThreshold"] =	simulation.sandThreshold;

	json["Simulation/SourceStrengths"] = { simulation.sourceStrengths.x, simulation.sourceStrengths.y, simulation.sourceStrengths.z, simulation.sourceStrengths.w };
	json["Simulation/SourceSize"] = { simulation.sourceSize.x, simulation.sourceSize.y, simulation.sourceSize.z, simulation.sourceSize.w };
	json["Simulation/SourceLocations"] = { 
		{ simulation.sourceLocations[0].x, simulation.sourceLocations[0].y }, 
		{ simulation.sourceLocations[1].x, simulation.sourceLocations[1].y }, 
		{ simulation.sourceLocations[2].x, simulation.sourceLocations[2].y }, 
		{ simulation.sourceLocations[3].x, simulation.sourceLocations[3].y }
	};


	const auto backgroundColor{ world.getSingleton<onec::AmbientLight>()->color };
	json["Rendering/VisualScale"] = world.getComponent<onec::Scale>(this->terrain.entity)->scale;
	json["Rendering/AmbientLight"] = world.getSingleton<onec::AmbientLight>()->strength;
	json["Rendering/BackgroundColor"] = { backgroundColor.r, backgroundColor.g, backgroundColor.b };
	json["Rendering/BedrockColor"] = { rendering.bedrockColor.r, rendering.bedrockColor.g, rendering.bedrockColor.b };
	json["Rendering/SandColor"] = { rendering.sandColor.r, rendering.sandColor.g, rendering.sandColor.b };
	json["Rendering/WaterColor"] = { rendering.waterColor.r, rendering.waterColor.g, rendering.waterColor.b };
	json["Rendering/UseInterpolation"] = rendering.useInterpolation;
	json["Rendering/RenderSand"] = rendering.renderSand;
	json["Rendering/RenderWater"] = rendering.renderWater;

	onec::writeFile(file, json.dump(1));
	onec::writeFile(file.stem().concat(".dat"), std::string_view{ reinterpret_cast<char*>(compressed.data()), compressed.size() });
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
	this->terrain.maxLayerCount = json["Terrain/MaxLayerCount"];

	terrain = Terrain{ this->terrain.gridSize, this->terrain.gridScale, static_cast<char>(terrain.maxLayerCount) };
	std::string compressed{ onec::readFile(file.stem().concat(".dat")) };

	const int usedLayerCount{ json["Terrain/UsedLayerCount"] };
	const std::ptrdiff_t cellCount{ terrain.gridSize.x * terrain.gridSize.y };
	const std::ptrdiff_t columnCount{ cellCount * usedLayerCount };
	const std::ptrdiff_t layerCountBytes{ cellCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::layerCounts)) };
	const std::ptrdiff_t heightBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::heights)) };
	const std::ptrdiff_t fluxBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::fluxes)) };
	const std::ptrdiff_t stabilityBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::stability)) };
	const std::ptrdiff_t sedimentBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::sediments)) };
	const std::ptrdiff_t damageBytes{ columnCount * static_cast<std::ptrdiff_t>(sizeof(*device::Simulation::damages)) };

	uLongf bytes{ 0 };
	bytes += layerCountBytes;
	bytes += heightBytes;
	bytes += fluxBytes;
	bytes += stabilityBytes;
	bytes += sedimentBytes;
	bytes += damageBytes;

	std::vector<std::byte> uncompressed(bytes);
	const int status{ uncompress(reinterpret_cast<Bytef*>(uncompressed.data()), &bytes, reinterpret_cast<const Bytef*>(compressed.data()), compressed.size()) };

	ONEC_ASSERT(status == Z_OK, "Failed to uncompress data");
	
	std::ptrdiff_t i{ 0 };
	terrain.layerCountBuffer.upload({ uncompressed.data() + i, layerCountBytes}); i += layerCountBytes;
	terrain.heightBuffer.upload({ uncompressed.data() + i, heightBytes }); i += heightBytes;
	terrain.fluxBuffer.upload({ uncompressed.data() + i, fluxBytes }); i += fluxBytes;
	terrain.stabilityBuffer.upload({ uncompressed.data() + i, stabilityBytes }); i += stabilityBytes;
	terrain.sedimentBuffer.upload({ uncompressed.data() + i, sedimentBytes }); i += sedimentBytes;
	terrain.damageBuffer.upload({ uncompressed.data() + i, damageBytes }); i += damageBytes;

	Simulation& simulation{ terrain.simulation };
	simulation.useOutflowBorders = json["Simulation/UseOutflowBorders"];
	simulation.deltaTime = json["Simulation/DeltaTime"];
	simulation.gravity = json["Simulation/Gravity"];
	simulation.rain = json["Simulation/Rain"];
	simulation.evaporation = json["Simulation/Evaporation"];
	simulation.petrification = json["Simulation/Petrification"];
	simulation.sedimentCapacityConstant = json["Simulation/SedimentCapacityConstant"];
	simulation.bedrockDissolvingConstant = json["Simulation/BedrockDissolvingConstant"];
	simulation.sandDissolvingConstant = json["Simulation/SandDissolvingConstant"];
	simulation.sedimentDepositionConstant = json["Simulation/SedimentDepositionConstant"];
	simulation.minSlopeErosionScale = json["Simulation/MinSlopeErosionScale"];
	simulation.maxSlopeErosionScale = json["Simulation/MaxSlopeErosionScale"];
	simulation.dryTalusAngle = json["Simulation/DryTalusAngle"];
	simulation.wetTalusAngle = json["Simulation/WetTalusAngle"];
	simulation.slippageInterpolationRange = json["Simulation/SlippageInterpolationRange"];
	simulation.minHorizontalErosionSlope = json["Simulation/MinHorizontalErosionSlope"];
	simulation.verticalErosionSlopeFadeStart = json["Simulation/VerticalErosionSlopeFadeStart"];
	simulation.horizontalErosionStrength = json["Simulation/HorizontalErosionStrength"];
	simulation.minSplitDamage = json["Simulation/MinSplitDamage"];
	simulation.splitSize = json["Simulation/SplitSize"];
	simulation.damageRecovery = json["Simulation/DamageRecovery"];
	simulation.verticalErosionEnabled = json["Simulation/EnableVerticalErosion"];
	simulation.horizontalErosionEnabled = json["Simulation/EnableHorizontalErosion"];
	simulation.slippageEnabled = json["Simulation/EnableSlippage"];
	simulation.supportCheckEnabled = json["Simulation/EnableSupportCheck"];
	simulation.useWeightInSupportCheck = json["Simulation/UseWeightInSupportCheck"];
	simulation.currentSimulationStep = json["Simulation/CurrentSimulationStep"];
	simulation.currentStabilityStep = json["Simulation/CurrentStabilityStep"];
	simulation.minBedrockThickness = json["Simulation/MinBedrockThickness"];
	simulation.erosionWaterMaxHeight = json["Simulation/ErosionWaterMaxHeight"];
	simulation.sandThreshold = json["Simulation/SandThreshold"];
	simulation.topErosionWaterScale = json["Simulation/TopErosionWaterScale"];
	simulation.evaporationEmptySpaceScale = json["Simulation/EvaporationEmptySpaceScale"];

	simulation.sourceStrengths = glm::vec4{
		json["Simulation/SourceStrengths"][0],
		json["Simulation/SourceStrengths"][1],
		json["Simulation/SourceStrengths"][2],
		json["Simulation/SourceStrengths"][3]
	};
	simulation.sourceSize = glm::vec4{
		json["Simulation/SourceSize"][0],
		json["Simulation/SourceSize"][1],
		json["Simulation/SourceSize"][2],
		json["Simulation/SourceSize"][3]
	};
	for (int i = 0; i < 4; ++i) {
		simulation.sourceLocations[i] = glm::ivec2{
			json["Simulation/SourceLocations"][i][0],
			json["Simulation/SourceLocations"][i][1]
		};
	}

	simulation.init = false;
	simulation.paused = true;

	world.getComponent<onec::Scale>(this->terrain.entity)->scale = json["Rendering/VisualScale"];
	world.getSingleton<onec::AmbientLight>()->color = glm::vec3{ json["Rendering/BackgroundColor"][0], json["Rendering/BackgroundColor"][1], json["Rendering/BackgroundColor"][2] };
	world.getSingleton<onec::AmbientLight>()->strength = json["Rendering/AmbientLight"];
	rendering.bedrockColor = glm::vec3{ json["Rendering/BedrockColor"][0], json["Rendering/BedrockColor"][1], json["Rendering/BedrockColor"][2] };
	rendering.sandColor = glm::vec3{ json["Rendering/SandColor"][0], json["Rendering/SandColor"][1], json["Rendering/SandColor"][2] };
	rendering.waterColor = glm::vec3{ json["Rendering/WaterColor"][0], json["Rendering/WaterColor"][1], json["Rendering/WaterColor"][2] };
	rendering.useInterpolation = json["Rendering/UseInterpolation"];
	rendering.renderSand = json["Rendering/RenderSand"];
	rendering.renderWater = json["Rendering/RenderWater"];

    PointRenderer& pointRenderer{ *onec::getWorld().getComponent<PointRenderer>(this->terrain.entity) };
	geo::SimpleMaterialUniforms uniforms;
	uniforms.bedrockColor = onec::sRGBToLinear(rendering.bedrockColor);
	uniforms.sandColor = onec::sRGBToLinear(rendering.sandColor);
	uniforms.waterColor = onec::sRGBToLinear(rendering.waterColor);
	uniforms.useInterpolation = rendering.useInterpolation;
	uniforms.renderSand = rendering.renderSand;
	uniforms.renderWater = rendering.renderWater;
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
