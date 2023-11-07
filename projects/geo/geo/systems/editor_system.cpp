#include "editor_system.hpp"
#include "../components/simulation.hpp"
#include "../singletons/editor.hpp"

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

	addDirectionalLight();
	addCamera(editor);
	addTerrain(editor);
	addMaterial(editor);
	addRenderMesh(editor);
	addSimulation(editor);

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

void EditorSystem::addDirectionalLight()
{
	onec::World& world{ onec::getWorld() };

	const entt::entity directionalLight{ world.addEntity() };
	world.addComponent<onec::DirectionalLight>(directionalLight);
	world.addComponent<onec::Rotation>(directionalLight, glm::quat{ glm::radians(glm::vec3{ -50.0f, 30.0f, 0.0f }) });
	world.addComponent<onec::LocalToWorld>(directionalLight);
	world.addComponent<onec::Static>(directionalLight);
}

void EditorSystem::addCamera(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	editor.camera.entity = world.addEntity();
	world.addComponent<onec::PerspectiveCamera>(editor.camera.entity, glm::radians(editor.camera.fieldOfView), editor.camera.nearPlane, editor.camera.farPlane);
	world.addComponent<onec::Trackball>(editor.camera.entity);
	world.addComponent<onec::Position>(editor.camera.entity, glm::vec3{ 0.0f, 0.0f, 5.0f });
	world.addComponent<onec::Rotation>(editor.camera.entity);
	world.addComponent<onec::LocalToWorld>(editor.camera.entity);
	world.addComponent<onec::WorldToView>(editor.camera.entity);
	world.addComponent<onec::ViewToClip>(editor.camera.entity);
	world.setSingleton<onec::ActiveCamera>(editor.camera.entity);
}

void EditorSystem::addTerrain(Editor& editor)
{
	editor.terrain.terrain = std::make_shared<geo::Terrain>();
	geo::Terrain& terrain{ *editor.terrain.terrain };

	terrain.heightMap.initialize(GL_TEXTURE_3D, glm::ivec3{ editor.terrain.gridSize, editor.terrain.maxLayerCount }, GL_RGBA32F);
	terrain.waterVelocityMap.initialize(GL_TEXTURE_3D, glm::ivec3{ editor.terrain.gridSize, editor.terrain.maxLayerCount }, GL_RG32F);
	terrain.indexMap.initialize(GL_TEXTURE_3D, glm::ivec3{ editor.terrain.gridSize, editor.terrain.maxLayerCount }, GL_RG32I);
}

void EditorSystem::addMaterial(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	editor.terrain.entity = world.addEntity();
	world.addComponent<onec::LocalToWorld>(editor.camera.entity);

	editor.rendering.material = std::make_shared<TerrainBRDF>();
	TerrainBRDF& material{ *editor.rendering.material };

	material.uniforms.gridSize = editor.terrain.gridSize;
	material.uniforms.maxLayerCount = editor.terrain.maxLayerCount;
	material.uniforms.bedrockColor = glm::vec4{ editor.rendering.bedrockColor, 1.0f };
	material.uniforms.sandColor = glm::vec4{ editor.rendering.sandColor, 1.0f };
	material.uniforms.waterColor = glm::vec4{ editor.rendering.waterColor, 1.0f };
	material.uniformBuffer.initialize(onec::asBytes(&material.uniforms, 1));

	const onec::Application& application{ onec::getApplication() };
	const std::filesystem::path assets{ application.getDirectory() / "assets" };

	material.program = std::make_shared<onec::Program>();
	material.program->attachShader(onec::Shader{ assets / "shaders/terrain/terrain.vert" });
	material.program->attachShader(onec::Shader{ assets / "shaders/terrain/terrain.geom" });
	material.program->attachShader(onec::Shader{ assets / "shaders/terrain/terrain.frag" });
	material.program->link();

	material.renderState = std::make_shared<onec::RenderState>();

	material.bindGroup = std::make_shared<onec::BindGroup>();
	material.bindGroup->attachTexture(0, editor.terrain.terrain->heightMap);
}

void EditorSystem::addRenderMesh(Editor& editor)
{
	onec::World& world{ onec::getWorld() };
	world.addComponent<onec::LocalToWorld>(editor.terrain.entity);

	const onec::Application& application{ onec::getApplication() };

	onec::RenderMesh& renderMesh{ world.addComponent<onec::RenderMesh>(editor.terrain.entity) };
	renderMesh.mesh = std::make_shared<onec::Mesh>(application.getDirectory() / "assets/meshes/cube.obj");
	renderMesh.materials.emplace_back(editor.rendering.material);
	renderMesh.instanceCount = 3 * editor.terrain.gridSize.x * editor.terrain.gridSize.y * editor.terrain.maxLayerCount;
}

void EditorSystem::addSimulation(Editor& editor)
{
	onec::World& world{ onec::getWorld() };

	Simulation& simulation{ world.addComponent<Simulation>(editor.terrain.entity) };
	simulation.terrain = editor.terrain.terrain;
	simulation.timeScale = editor.simulation.timeScale;
	simulation.gravityScale = editor.simulation.gravityScale;
	simulation.isPaused = editor.simulation.isPaused;
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

		if (ImGui::DragInt("Target Frame Rate", &editor.application.targetFramerate, 1.0f, 30, 3000))
		{
			application.setTargetFrameRate(editor.application.targetFramerate);
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

		if (ImGui::DragFloat("Field Of View", &editor.camera.fieldOfView, 0.367f, 173.0f))
		{
			perspectiveCamera.fieldOfView = glm::radians(editor.camera.fieldOfView);
		}

		if (ImGui::DragFloat("Near Plane", &editor.camera.nearPlane, 1.0f, 0.001f))
		{
			perspectiveCamera.nearPlane = editor.camera.nearPlane;
		}

		if (ImGui::DragFloat("Far Plane", &editor.camera.farPlane, 1.0f, 0.001f))
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
		if (ImGui::DragInt2("Grid Size", &editor.terrain.gridSize.x, 1.0f, 16, 4096))
		{

		}

		if (ImGui::DragFloat("Grid Scale", &editor.terrain.gridScale, 1.0f, 0.001f))
		{

		}

		if (ImGui::DragInt("Max Layer Count", &editor.terrain.maxLayerCount, 1.0f, 1))
		{

		}

		ImGui::TreePop();
	}
}

void EditorSystem::updateSimulationGUI(Editor& editor)
{
	if (ImGui::TreeNode("Simulation"))
	{
		onec::World& world{ onec::getWorld() };

		ONEC_ASSERT(world.hasComponent<Simulation>(editor.terrain.entity), "Terrain must have a perspective camera component");

		Simulation& simulation{ *world.getComponent<Simulation>(editor.terrain.entity) };

		if (ImGui::DragFloat("Time Scale", &editor.simulation.timeScale))
		{
			simulation.timeScale = editor.simulation.timeScale;
		}

		if (ImGui::DragFloat("Gravity Scale", &editor.simulation.gravityScale))
		{
			simulation.gravityScale = editor.simulation.gravityScale;
		}

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
