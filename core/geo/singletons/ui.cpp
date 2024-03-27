#include "ui.hpp"
#include "imgui.h"
#include "../components/terrain.hpp"
#include "../resources/simple_material.hpp"
#include <onec/onec.hpp>

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

	updateApplication();
	updateCamera();
	updateTerrain();
	updateSimulation();
	updateRendering();

	ImGui::End();
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

			world.setComponent<onec::Position>(entity, -0.5f * uniforms.gridScale * world.getComponent<onec::Scale>(entity)->scale * glm::vec3{ uniforms.gridSize.x, 0.0f, uniforms.gridSize.y });

			onec::MeshRenderer& meshRenderer{ *world.getComponent<onec::MeshRenderer>(entity) };
			meshRenderer.materials[0]->uniformBuffer.upload(onec::asBytes(&uniforms, 1));
			meshRenderer.instanceCount = terrain.maxLayerCount * uniforms.gridSize.x * uniforms.gridSize.y;
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

		ImGui::DragFloat("Sediment Capacity Constant", &simulation.sedimentCapacityConstant, 0.001f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Dissolving Constant", &simulation.dissolvingConstant, 0.001f, 0.0f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Deposition Constant", &simulation.depositionConstant, 0.001f, 0.0f, std::numeric_limits<float>::max());

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
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ terrain.entity };

		Terrain& terrain{ *world.getComponent<Terrain>(entity) };
		float& scale{ world.getComponent<onec::Scale>(entity)->scale };
		onec::MeshRenderer& meshRenderer{ *world.getComponent<onec::MeshRenderer>(entity) };
		
		if (ImGui::DragFloat("Visual Scale", &scale, 0.1f, 0.001f, std::numeric_limits<float>::max()))
		{
			world.setComponent<onec::Position>(entity, -0.5f * terrain.gridScale * scale * glm::vec3{ terrain.gridSize.x, 0.0f, terrain.gridSize.y });
		}

		ImGui::DragFloat("Ambient Light [lux]", &world.getSingleton<onec::AmbientLight>()->strength, 1.0f, 0.0f, std::numeric_limits<float>::max());
		ImGui::ColorEdit3("Background Color", &world.getSingleton<onec::RenderPipeline>()->clearColor.x);

		if (ImGui::ColorEdit3("Bedrock Color", &rendering.bedrockColor.x))
		{
			const glm::vec3 bedrockColor{ onec::sRGBToLinear(rendering.bedrockColor) };
			meshRenderer.materials[0]->uniformBuffer.upload(onec::asBytes(&bedrockColor, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, bedrockColor)), sizeof(glm::vec3));
		}

		if (ImGui::ColorEdit3("Sand Color", &rendering.sandColor.x))
		{
			const glm::vec3 sandColor{ onec::sRGBToLinear(rendering.sandColor) };
			meshRenderer.materials[0]->uniformBuffer.upload(onec::asBytes(&sandColor, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, sandColor)), sizeof(glm::vec3));
		}

		if (ImGui::ColorEdit3("Water Color", &rendering.waterColor.x))
		{
			const glm::vec3 waterColor{ onec::sRGBToLinear(rendering.waterColor) };
			meshRenderer.materials[0]->uniformBuffer.upload(onec::asBytes(&waterColor, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, waterColor)), sizeof(glm::vec3));
		}

		bool useInterpolation = bool(rendering.useInterpolation);
		if (ImGui::Checkbox("Use Interpolation", &useInterpolation)) {
			rendering.useInterpolation = int(useInterpolation);
			meshRenderer.materials[0]->uniformBuffer.upload(onec::asBytes(&rendering.useInterpolation, 1), static_cast<std::ptrdiff_t>(offsetof(SimpleMaterialUniforms, useInterpolation)), sizeof(int));
		}

		ImGui::TreePop();
	}
}

}
