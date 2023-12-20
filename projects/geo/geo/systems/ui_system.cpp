#include "ui_system.hpp"
#include "../components/simulation.hpp"
#include "../singletons/ui.hpp"
#include "../graphics/material.hpp"
#include <imgui.h>
#include <limits>

namespace geo
{

void updateApplicationUI(UI& ui)
{
	onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };

	if (ImGui::TreeNode("Application"))
	{
		int targetFrameRate{ application.getTargetFrameRate() };
		float fixedDeltaTime{ static_cast<float>(application.getFixedDeltaTime()) };
		bool isVSyncEnabled{ window.getSwapInterval() != 0 };

		if (ImGui::DragInt("Target Frame Rate", &targetFrameRate, 0.5f, 30, 3000))
		{
			application.setTargetFrameRate(targetFrameRate);
		}

		if (ImGui::DragFloat("Fixed Delta Time", &fixedDeltaTime, 0.01f, 0.001f, 2.0f, "%.3fs"))
		{
			application.setFixedDeltaTime(fixedDeltaTime);
		}

		if (ImGui::Checkbox("Vertical Synchronization", &isVSyncEnabled))
		{
			window.setSwapInterval(isVSyncEnabled);
		}

		ImGui::TreePop();
	}
}

void updateCameraUI(UI& ui)
{
	if (ImGui::TreeNode("Camera"))
	{
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ ui.camera.entity };

		ONEC_ASSERT(world.hasComponent<onec::PerspectiveCamera>(entity), "Camera must have a perspective camera component");

		onec::PerspectiveCamera& perspectiveCamera{ *world.getComponent<onec::PerspectiveCamera>(entity) };

		float fieldOfView{ glm::degrees(perspectiveCamera.fieldOfView) };

		if (ImGui::DragFloat("Field Of View", &fieldOfView, 0.1f, 0.367f, 173.0f, "%.3fdeg"))
		{
			perspectiveCamera.fieldOfView = glm::radians(fieldOfView);
		}

		ImGui::DragFloat("Near Plane", &perspectiveCamera.nearPlane, 0.01f, 0.001f, std::numeric_limits<float>::max(), "%.3fm");
		ImGui::DragFloat("Far Plane", &perspectiveCamera.farPlane, 0.5f, 0.001f, std::numeric_limits<float>::max(), "%.3fm");

		ImGui::TreePop();
	}
}

void updateTerrainUI(UI& ui)
{
	if (ImGui::TreeNode("Terrain"))
	{
		if (ImGui::Button("Reset"))
		{
			onec::World& world{ onec::getWorld() };

			const entt::entity entity{ ui.terrain.entity };

			ONEC_ASSERT(world.hasComponent<onec::RenderMesh>(entity), "Terrain must have a render mesh component");

			const glm::ivec3 gridSize{ ui.terrain.gridSize };
			const float gridScale{ ui.terrain.gridScale };

			const std::shared_ptr<geo::Terrain> terrain{ std::make_shared<geo::Terrain>(gridSize, gridScale) };
			const std::shared_ptr<geo::Material> material{ std::make_shared<geo::Material>(*terrain) };

			world.setComponent<geo::Simulation>(entity, terrain);

			onec::RenderMesh& renderMesh{ *world.getComponent<onec::RenderMesh>(entity) };
			renderMesh.materials[0] = material;
			renderMesh.instanceCount = gridSize.x * gridSize.y * gridSize.z;
		}

		ImGui::DragInt2("Grid Size", &ui.terrain.gridSize.x, 0.5f, 16, 4096);
		ImGui::DragFloat("Grid Scale", &ui.terrain.gridScale, 0.01f, 0.001f, 10.0f, "%.3fm");
		ImGui::DragInt("Max Layer Count", &ui.terrain.gridSize.z, 0.5f, 1, std::numeric_limits<int>::max());

		ImGui::TreePop();
	}
}

void updateSimulationUI(UI& ui)
{
	if (ImGui::TreeNode("Simulation"))
	{
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ ui.terrain.entity };

		ONEC_ASSERT(world.hasComponent<Simulation>(entity), "Terrain must have a simulation component");
		ONEC_ASSERT(world.hasSingleton<onec::Gravity>(), "World must have a gravity singleton");

		const bool isPaused{ world.hasComponent<onec::Disabled<Simulation>>(entity) };

		if (isPaused)
		{
			if (ImGui::Button("Start"))
			{
				world.removeComponent<onec::Disabled<Simulation>>(entity);
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
			{
				world.addComponent<onec::Disabled<Simulation>>(entity);
			}
		}

		Simulation& simulation{ *world.getComponent<Simulation>(entity) };

		ImGui::DragFloat("Gravity", &world.getSingleton<onec::Gravity>()->gravity.y, 0.1f, 0.0f, 0.0f, "%.3fm/s^2");
		ImGui::DragFloat("Rain", &simulation.data.rain, 0.01f, 0.0f, std::numeric_limits<float>::max(), "%.3fm/(m^2 * s)");

		float evaporation{ 100.0f * simulation.data.evaporation };

		if (ImGui::DragFloat("Evaporation", &evaporation, 0.5f, 0.0f, std::numeric_limits<float>::max(), "%.3f%%/s"))
		{
			simulation.data.evaporation = 0.01f * evaporation;
		}
		
		ImGui::TreePop();
	}
}

void updateRenderingUI(UI& ui)
{
	if (ImGui::TreeNode("Rendering"))
	{
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ ui.terrain.entity };

		ONEC_ASSERT(world.hasComponent<onec::Scale>(entity), "Terrain must have a scale component");
		ONEC_ASSERT(world.hasComponent<onec::RenderMesh>(entity), "Terrain must have a render mesh component");
		ONEC_ASSERT(world.hasSingleton<onec::Renderer>(), "Terrain must have a renderer singleton");

		const bool isPaused{ world.hasComponent<onec::Disabled<onec::RenderMesh>>(entity) };

		if (isPaused)
		{
			if (ImGui::Button("Start"))
			{
				world.removeComponent<onec::Disabled<onec::RenderMesh>>(entity);
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
			{
				world.addComponent<onec::Disabled<onec::RenderMesh>>(entity);
			}
		}

		ImGui::DragFloat("Visual Scale", &world.getComponent<onec::Scale>(entity)->scale, 0.1f, 0.001f, std::numeric_limits<float>::max());
		ImGui::ColorEdit4("Background Color", &world.getSingleton<onec::Renderer>()->clearColor.x);

		geo::Material& material{ *reinterpret_cast<geo::Material*>(world.getComponent<onec::RenderMesh>(entity)->materials[0].get()) };
		bool materialHasChanged{ false };
		materialHasChanged |= ImGui::ColorEdit3("Bedrock Color", &material.uniforms.bedrockColor.x);
		materialHasChanged |= ImGui::ColorEdit3("Sand Color", &material.uniforms.sandColor.x);
		materialHasChanged |= ImGui::ColorEdit3("Water Color", &material.uniforms.waterColor.x);

		if (materialHasChanged)
		{
			material.uniformBuffer.initialize(onec::asBytes(&material.uniforms, 1));
		}

		ImGui::TreePop();
	}
}

void updateUI()
{
	const onec::Application& application{ onec::getApplication() };
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<UI>(), "World must have a display singleton");

	const std::string fps{ std::to_string(application.getFrameRate()) + "fps" };
	const std::string ms{ std::to_string(1000.0 * application.getUnscaledDeltaTime()) + "ms" };
	const std::string title{ application.getName() + " @ " + fps + " / " + ms };
	window.setTitle(title);

	geo::UI& ui{ *world.getSingleton<UI>() };
	const bool isVisable{ ui.isVisable != window.isKeyPressed(GLFW_KEY_ESCAPE) };

	ui.isVisable = isVisable;
	
	if (!isVisable)
	{
		return;
	}

	ImGui::Begin("UI", &ui.isVisable);

	updateApplicationUI(ui);
	updateCameraUI(ui);
	updateTerrainUI(ui);
	updateSimulationUI(ui);
	updateRenderingUI(ui);

	ImGui::End();
}

}
