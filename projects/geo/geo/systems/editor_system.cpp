#include "editor_system.hpp"
#include "../components/simulation.hpp"
#include "../singletons/editor.hpp"
#include "../systems/simulation_system.hpp"

namespace geo
{

void EditorSystem::start()
{
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<Editor>(), "World must have an editor singleton");

	geo::Editor& editor{ *world.getSingleton<Editor>() };

	onec::Application& application{ onec::getApplication() };
	application.setTargetFrameRate(editor.application.targetFramerate);
	application.setVSyncCount(editor.application.isVSyncEnabled);
	application.setTimeScale(editor.application.timeScale);
	application.setFixedDeltaTime(editor.application.fixedDeltaTime);

	initializeCamera(editor);
	initializeTerrain(editor);
	initializeMaterial(editor);
	initializeRenderMesh(editor);
	initializeSimulation(editor);

	onec::Renderer& renderer{ *world.getSingleton<onec::Renderer>() };
	renderer.clearColor = glm::vec4{ editor.rendering.backgroundColor, 1.0f };
}

void EditorSystem::update()
{
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<Editor>(), "World must have an editor singleton");

	geo::Editor& editor{ *world.getSingleton<Editor>() };

	ImGui::Begin("Editor");

	updateApplicationGUI(editor);
	updateCameraGUI(editor);
	updateTerrainGUI(editor);
	updateSimulationGUI(editor);
	updateRenderingGUI(editor);

	ImGui::End();
}

void EditorSystem::initializeCamera(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	editor.camera.entity = world.addEntity();

	const float maxLength{ editor.terrain.gridScale * static_cast<float>(glm::max(editor.terrain.gridSize.x, editor.terrain.gridSize.y)) };

	world.addComponent<onec::PerspectiveCamera>(editor.camera.entity, glm::radians(editor.camera.fieldOfView), editor.camera.nearPlane, editor.camera.farPlane);
	world.addComponent<onec::Trackball>(editor.camera.entity);
	world.addComponent<onec::Position>(editor.camera.entity, glm::vec3{ 0.0f, 0.25f * maxLength, 0.75f * maxLength });
	world.addComponent<onec::Rotation>(editor.camera.entity);
	world.addComponent<onec::LocalToWorld>(editor.camera.entity);
	world.addComponent<onec::WorldToView>(editor.camera.entity);
	world.addComponent<onec::ViewToClip>(editor.camera.entity);

	world.setSingleton<onec::ActiveCamera>(editor.camera.entity);
}

void EditorSystem::initializeTerrain(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	editor.terrain.terrain = std::make_shared<geo::Terrain>();
	editor.terrain.entity = world.addEntity();

	updateTerrain(editor);
}

void EditorSystem::initializeMaterial(Editor& editor)
{
	editor.rendering.material = std::make_shared<TerrainBRDF>();
	TerrainBRDF& material{ *editor.rendering.material };

	const onec::Application& application{ onec::getApplication() };
	const std::filesystem::path assets{ application.getDirectory() / "assets" };

	material.program = std::make_shared<onec::Program>();
	material.program->attachShader(onec::Shader{ assets / "shaders/terrain/terrain.vert" });
	material.program->attachShader(onec::Shader{ assets / "shaders/terrain/terrain.geom" });
	material.program->attachShader(onec::Shader{ assets / "shaders/terrain/terrain.frag" });
	material.program->link();

	material.renderState = std::make_shared<onec::RenderState>();
	material.bindGroup = std::make_shared<onec::BindGroup>();

	updateMaterial(editor);
}

void EditorSystem::initializeRenderMesh(Editor& editor)
{
	const onec::Application& application{ onec::getApplication() };

	onec::World& world{ onec::getWorld() };
	world.addComponent<onec::LocalToWorld>(editor.terrain.entity);

	onec::RenderMesh& renderMesh{ world.addComponent<onec::RenderMesh>(editor.terrain.entity) };
	renderMesh.mesh = std::make_shared<onec::Mesh>(application.getDirectory() / "assets/meshes/cube.obj");
	renderMesh.materials.emplace_back(editor.rendering.material);

	updateRenderMesh(editor);
}

void EditorSystem::initializeSimulation(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	Simulation& simulation{ world.addComponent<Simulation>(editor.terrain.entity) };
	simulation.terrain = editor.terrain.terrain;
	simulation.isPaused = editor.simulation.isPaused;

	SimulationSystem::initialize(simulation);
}

void EditorSystem::updateTerrain(Editor& editor)
{
	geo::Terrain& terrain{ *editor.terrain.terrain };
	terrain.infoMap.initialize(GL_TEXTURE_3D, glm::ivec3{ editor.terrain.gridSize, editor.terrain.maxLayerCount }, GL_RGBA8I);
	terrain.heightMap.initialize(GL_TEXTURE_3D, glm::ivec3{ editor.terrain.gridSize, editor.terrain.maxLayerCount }, GL_RGBA32F);
	terrain.waterVelocityMap.initialize(GL_TEXTURE_3D, glm::ivec3{ editor.terrain.gridSize, editor.terrain.maxLayerCount }, GL_RG32F);
}

void EditorSystem::updateMaterial(Editor& editor)
{
	TerrainBRDF& material{ *editor.rendering.material };

	material.uniforms.gridSize = editor.terrain.gridSize;
	material.uniforms.gridScale = editor.terrain.gridScale;
	material.uniforms.maxLayerCount = editor.terrain.maxLayerCount;
	material.uniforms.bedrockColor = glm::vec4{ editor.rendering.bedrockColor, 1.0f };
	material.uniforms.sandColor = glm::vec4{ editor.rendering.sandColor, 1.0f };
	material.uniforms.waterColor = glm::vec4{ editor.rendering.waterColor, 1.0f };
	material.uniformBuffer.initialize(onec::asBytes(&material.uniforms, 1));

	material.bindGroup->attachTexture(0, editor.terrain.terrain->infoMap);
	material.bindGroup->attachTexture(1, editor.terrain.terrain->heightMap);
	material.bindGroup->attachTexture(2, editor.terrain.terrain->waterVelocityMap);
}

void EditorSystem::updateRenderMesh(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasComponent<onec::RenderMesh>(editor.terrain.entity), "Terrain must have a render mesh component");

	world.setComponent<onec::Position>(editor.terrain.entity, glm::vec3{ -0.5f * editor.terrain.gridScale * editor.terrain.gridSize.x, 0.0f, -0.5f * editor.terrain.gridScale * editor.terrain.gridSize.y });

	onec::RenderMesh& renderMesh{ *world.getComponent<onec::RenderMesh>(editor.terrain.entity) };
	renderMesh.instanceCount = editor.terrain.gridSize.x * editor.terrain.gridSize.y * editor.terrain.maxLayerCount;
}

void EditorSystem::updateApplicationGUI(Editor& editor)
{
	onec::Application& application{ onec::getApplication() };

	if (ImGui::TreeNode("Application"))
	{
		if (ImGui::Checkbox("Vertical Synchronization", &editor.application.isVSyncEnabled))
		{
			application.setVSyncCount(editor.application.isVSyncEnabled);
		}

		if (ImGui::DragInt("Target Frame Rate", &editor.application.targetFramerate))
		{
			application.setTargetFrameRate(editor.application.targetFramerate);
		}

		if (ImGui::DragFloat("Time Scale", &editor.application.timeScale))
		{
			application.setTimeScale(editor.application.timeScale);
		}

		if (ImGui::DragFloat("Fixed Delta Time [s]", &editor.application.fixedDeltaTime))
		{
			application.setFixedDeltaTime(editor.application.fixedDeltaTime);
		}

		ImGui::TreePop();
	}
}

void EditorSystem::updateCameraGUI(Editor& editor)
{
	if (ImGui::TreeNode("Camera"))
	{
		onec::World& world{ onec::getWorld() };

		ONEC_ASSERT(world.hasComponent<onec::PerspectiveCamera>(editor.camera.entity), "Camera must have a perspective camera component");

		onec::PerspectiveCamera& perspectiveCamera{ *world.getComponent<onec::PerspectiveCamera>(editor.camera.entity) };

		if (ImGui::DragFloat("Field Of View [°]", &editor.camera.fieldOfView))
		{
			perspectiveCamera.fieldOfView = glm::radians(editor.camera.fieldOfView);
		}

		if (ImGui::DragFloat("Near Plane [m]", &editor.camera.nearPlane))
		{
			perspectiveCamera.nearPlane = editor.camera.nearPlane;
		}

		if (ImGui::DragFloat("Far Plane [m]", &editor.camera.farPlane))
		{
			perspectiveCamera.farPlane = editor.camera.farPlane;
		}

		ImGui::TreePop();
	}
}

void EditorSystem::updateTerrainGUI(Editor& editor)
{
	if (ImGui::TreeNode("Terrain"))
	{
		if (ImGui::Button("Reset"))
		{
			updateTerrain(editor);
			updateMaterial(editor);
			updateRenderMesh(editor);
		}

		ImGui::DragInt2("Grid Size", &editor.terrain.gridSize.x);
		ImGui::DragFloat("Grid Scale [m]", &editor.terrain.gridScale);
		ImGui::DragInt("Max Layer Count", &editor.terrain.maxLayerCount);



		ImGui::TreePop();
	}
}

void EditorSystem::updateSimulationGUI(Editor& editor)
{
	if (ImGui::TreeNode("Simulation"))
	{
		onec::Application& application{ onec::getApplication() };
		onec::World& world{ onec::getWorld() };

		ONEC_ASSERT(world.hasComponent<Simulation>(editor.terrain.entity), "Terrain must have a perspective camera component");

		Simulation& simulation{ *world.getComponent<Simulation>(editor.terrain.entity) };

		updateRainGUI(editor, simulation);

		ImGui::TreePop();
	}
}

void EditorSystem::updateRainGUI(Editor& editor, Simulation& simulation)
{
	if (ImGui::TreeNode("Rain"))
	{
		if (editor.simulation.rain.isPaused)
		{
			if (ImGui::Button("Start"))
			{
				editor.simulation.rain.isPaused = false;
				simulation.deviceParameters.rain.amount = editor.simulation.rain.amount;
			}
		}
		else
		{
			if (ImGui::Button("Stop"))
			{
				editor.simulation.rain.isPaused = true;
				simulation.deviceParameters.rain.amount = 0.0f;
			}
		}

		ImGui::DragFloat("Amount [mm/m^2 * s]", &editor.simulation.rain.amount);

		ImGui::TreePop();
	}
}

void EditorSystem::updateRenderingGUI(Editor& editor)
{
	if (ImGui::TreeNode("Rendering"))
	{
		onec::World& world{ onec::getWorld() };

		if (ImGui::ColorEdit3("Background Color", &editor.rendering.backgroundColor.x))
		{
			ONEC_ASSERT(world.hasSingleton<onec::Renderer>(), "World must have a renderer singleton");

			onec::Renderer& renderer{ *world.getSingleton<onec::Renderer>() };
			renderer.clearColor = glm::vec4{ editor.rendering.backgroundColor, 1.0f };
		}

		bool materialHasChanged{ static_cast<bool>(ImGui::ColorEdit3("Bedrock Color", &editor.rendering.bedrockColor.x) |
								                   ImGui::ColorEdit3("Sand Color", &editor.rendering.sandColor.x) |
												   ImGui::ColorEdit3("Water Color", &editor.rendering.waterColor.x)) };

		if (materialHasChanged)
		{
			TerrainBRDF& material{ *editor.rendering.material };
			material.uniforms.bedrockColor = glm::vec4{ editor.rendering.bedrockColor, 1.0f };
			material.uniforms.sandColor = glm::vec4{ editor.rendering.sandColor, 1.0f };
			material.uniforms.waterColor = glm::vec4{ editor.rendering.waterColor, 1.0f };
			material.uniformBuffer.initialize(onec::asBytes(&material.uniforms, 1));
		}

		ImGui::TreePop();
	}
}

}
