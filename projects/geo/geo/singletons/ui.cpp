#include "ui.hpp"
#include "../components/simulation.hpp"
#include "../graphics/material.hpp"
#include <imgui.h>
#include <limits>

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
		float fixedDeltaTime{ static_cast<float>(application.getFixedDeltaTime()) };
		bool vSync{ window.getSwapInterval() != 0 };

		if (ImGui::DragInt("Target Frame Rate", &targetFrameRate, 0.5f, 30, 3000))
		{
			application.setTargetFrameRate(targetFrameRate);
		}

		if (ImGui::DragFloat("Fixed Delta Time [s]", &fixedDeltaTime, 0.01f, 0.001f, std::numeric_limits<float>::max()))
		{
			application.setFixedDeltaTime(fixedDeltaTime);
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

		ONEC_ASSERT(world.hasComponent<onec::PerspectiveCamera>(entity), "Camera must have a perspective camera component");

		onec::PerspectiveCamera& perspectiveCamera{ *world.getComponent<onec::PerspectiveCamera>(entity) };

		float fieldOfView{ glm::degrees(perspectiveCamera.fieldOfView) };

		if (ImGui::DragFloat("Field Of View [deg]", &fieldOfView, 0.1f, 0.367f, 173.0f))
		{
			perspectiveCamera.fieldOfView = glm::radians(fieldOfView);
		}

		ImGui::DragFloat("Near Plane [m]", &perspectiveCamera.nearPlane, 0.01f, 0.001f, std::numeric_limits<float>::max());
		ImGui::DragFloat("Far Plane [m]", &perspectiveCamera.farPlane, 0.5f, 0.001f, std::numeric_limits<float>::max());

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

			ONEC_ASSERT(world.hasComponent<onec::MeshRenderer>(entity), "Terrain must have a render mesh component");

			const glm::ivec3 gridSize{ terrain.gridSize };
			const float gridScale{ terrain.gridScale };

			const std::shared_ptr<geo::Terrain> terrain{ std::make_shared<geo::Terrain>(gridSize, gridScale) };
			const std::shared_ptr<geo::Material> material{ std::make_shared<geo::Material>(*terrain) };

			world.setComponent<geo::Simulation>(entity, terrain);

			onec::MeshRenderer& meshRenderer{ *world.getComponent<onec::MeshRenderer>(entity) };
			meshRenderer.materials[0] = material;
			meshRenderer.instanceCount = gridSize.x * gridSize.y * gridSize.z;
		}

		ImGui::DragInt2("Grid Size", &terrain.gridSize.x, 0.5f, 16, 4096);
		ImGui::DragFloat("Grid Scale [m]", &terrain.gridScale, 0.01f, 0.001f, 10.0f);
		ImGui::DragInt("Max Layer Count", &terrain.gridSize.z, 0.5f, 1, std::numeric_limits<int>::max());

		ImGui::TreePop();
	}
}

void UI::updateSimulation()
{
	if (ImGui::TreeNode("Simulation"))
	{
		onec::World& world{ onec::getWorld() };
		const entt::entity entity{ terrain.entity };

		ONEC_ASSERT(world.hasComponent<Simulation>(entity), "Terrain must have a simulation component");
		ONEC_ASSERT(world.hasSingleton<onec::Gravity>(), "World must have a gravity singleton");

		const bool isPaused{ world.hasComponent<onec::Inactive<Simulation>>(entity) };

		if (isPaused)
		{
			if (ImGui::Button("Start"))
			{
				world.removeComponent<onec::Inactive<Simulation>>(entity);
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
			{
				world.addComponent<onec::Inactive<Simulation>>(entity);
			}
		}

		Simulation& simulation{ *world.getComponent<Simulation>(entity) };

		ImGui::DragFloat("Gravity [m/s^2]", &world.getSingleton<onec::Gravity>()->gravity.y, 0.1f, 0.0f, 0.0f);
		ImGui::DragFloat("Rain [m/(m^2 * s)]", &simulation.data.rain, 0.01f, 0.0f, std::numeric_limits<float>::max());

		float evaporation{ 100.0f * simulation.data.evaporation };

		if (ImGui::DragFloat("Evaporation [%/s]", &evaporation, 0.5f, 0.0f, std::numeric_limits<float>::max()))
		{
			simulation.data.evaporation = 0.01f * evaporation;
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

		ONEC_ASSERT(world.hasComponent<onec::Transform>(entity), "Terrain must have a transform component");
		ONEC_ASSERT(world.hasComponent<onec::MeshRenderer>(entity), "Terrain must have a mesh renderer component");
		ONEC_ASSERT(world.hasSingleton<onec::RenderPipeline>(), "Terrain must have a render pipeline singleton");

		const bool isPaused{ world.hasComponent<onec::Inactive<onec::MeshRenderer>>(entity) };

		if (isPaused)
		{
			if (ImGui::Button("Start"))
			{
				world.removeComponent<onec::Inactive<onec::MeshRenderer>>(entity);
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
			{
				world.addComponent<onec::Inactive<onec::MeshRenderer>>(entity);
			}
		}

		ImGui::DragFloat("Visual Scale", &world.getComponent<onec::Transform>(entity)->scale, 0.1f, 0.001f, std::numeric_limits<float>::max());
		ImGui::ColorEdit4("Background Color", &world.getSingleton<onec::RenderPipeline>()->clearColor.x);

		geo::Material& material{ *reinterpret_cast<geo::Material*>(world.getComponent<onec::MeshRenderer>(entity)->materials[0].get()) };
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

}
