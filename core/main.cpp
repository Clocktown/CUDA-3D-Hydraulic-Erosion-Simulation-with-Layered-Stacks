#include "geo/geo.hpp"
#include <onec/onec.hpp>

void start()
{
	onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };

	world.addSingleton<onec::Viewport>();
	world.addSingleton<onec::AmbientLight>();
	world.addSingleton<onec::RenderPipeline>().clearColor = glm::vec4{ 0.7f, 0.9f, 1.0f, 1.0f };
	world.addSingleton<onec::MeshRenderPipeline>();

	geo::UI& ui{ world.addSingleton<geo::UI>() };
	const glm::ivec2 gridSize{ ui.terrain.gridSize };
	const float gridScale{ ui.terrain.gridScale };

	{
		const entt::entity entity{ world.addEntity() };
		const float maxLength{ gridScale * glm::max(gridSize.x, gridSize.y) };

		world.addComponent<onec::PerspectiveCamera>(entity, glm::radians(60.0f), 0.1f, 1000.0f);
		world.addComponent<onec::Trackball>(entity);
		world.addComponent<onec::Position>(entity, glm::vec3{ 0.0f, 0.25f * maxLength, 0.75f * maxLength });
		world.addComponent<onec::Rotation>(entity);
		world.addComponent<onec::WorldToView>(entity);
		world.addComponent<onec::ViewToClip>(entity);
		world.addComponent<onec::Exposure>(entity);

		world.addSingleton<onec::ActiveCamera>(entity);

		ui.camera.entity = entity;
	}

	{
		const entt::entity entity{ world.addEntity() };

		world.addComponent<onec::DirectionalLight>(entity);
		world.addComponent<onec::Position>(entity);
		world.addComponent<onec::Rotation>(entity, glm::quat{ glm::radians(glm::vec3{ -50.0f, 30.0f, 0.0f }) });
		world.addComponent<onec::Scale>(entity);
		world.addComponent<onec::LocalToWorld>(entity);
		world.addComponent<onec::Static>(entity);
	}

	{
		const entt::entity entity{ world.addEntity() };

		world.addComponent<onec::Position>(entity, -0.5f * gridScale * glm::vec3{ gridSize.x, 0.0f, gridSize.y });
		world.addComponent<onec::Rotation>(entity);
		world.addComponent<onec::Scale>(entity);
		world.addComponent<onec::LocalToWorld>(entity);

		geo::Terrain& terrain{ world.addComponent<geo::Terrain>(entity, gridSize, gridScale) };
		geo::SimpleMaterialUniforms uniforms;
		uniforms.bedrockColor = onec::sRGBToLinear(ui.rendering.bedrockColor);
		uniforms.sandColor = onec::sRGBToLinear(ui.rendering.sandColor);
		uniforms.waterColor = onec::sRGBToLinear(ui.rendering.waterColor);
		uniforms.gridSize = gridSize;
		uniforms.gridScale = gridScale;
		uniforms.maxLayerCount = terrain.maxLayerCount;
		uniforms.layerCounts = terrain.layerCountBuffer.getBindlessHandle();
		uniforms.heights = terrain.heightBuffer.getBindlessHandle();

		const std::filesystem::path assets{ application.getDirectory() / "assets" };
		const auto mesh{ std::make_shared<onec::Mesh>(assets / "meshes/cube.obj") };
		const auto material{ std::make_shared<onec::Material>() };
		const std::array<std::filesystem::path, 3> shaders{ assets / "shaders/simple_terrain/simple_terrain.vert",
															assets / "shaders/simple_terrain/simple_terrain.geom",
															assets / "shaders/simple_terrain/simple_terrain.frag" };

		material->program = std::make_shared<onec::Program>(shaders);
		material->renderState = std::make_shared<onec::RenderState>();
		material->uniformBuffer.initialize(onec::asBytes(&uniforms, 1));

		onec::MeshRenderer& meshRenderer{ world.addComponent<onec::MeshRenderer>(entity) };
		meshRenderer.mesh = mesh;
		meshRenderer.materials.emplace_back(material);
		meshRenderer.instanceCount = terrain.maxLayerCount * gridSize.x * gridSize.y;

		ui.terrain.entity = entity;
	}
	
	onec::updateModelMatrices();
	onec::updateViewMatrices();
	onec::updateProjectionMatrices();
}

void update()
{
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	if (!window.isMinimized())
	{
		world.getSingleton<onec::Viewport>()->size = window.getFramebufferSize();
	}

	geo::UI& ui{ *world.getSingleton<geo::UI>() };
	ui.update();

	geo::updateTerrains();
	onec::updateTrackballs();
	onec::updateModelMatrices(entt::exclude<onec::Static>);
	onec::updateViewMatrices(entt::exclude<onec::Static>);
	onec::updateProjectionMatrices();

	onec::RenderPipeline& renderPipeline{ *world.getSingleton<onec::RenderPipeline>() };
	renderPipeline.update();
	renderPipeline.render();
}

void render()
{
	onec::World& world{ onec::getWorld() };
	onec::MeshRenderPipeline& meshRenderPipeline{ *world.getSingleton<onec::MeshRenderPipeline>() };
	meshRenderPipeline.render();
}

int main()
{
	onec::Application& application{ onec::createApplication("Hydraulic Erosion", glm::ivec2{ 1280, 720 }, 0) };
	application.setTargetFrameRate(3000);

	onec::Window& window{ onec::getWindow() };
	window.setSwapInterval(0);

	onec::World& world{ onec::getWorld() };
	world.addSystem<onec::OnStart, &start>();
	world.addSystem<onec::OnUpdate, &update>();
	world.addSystem<onec::OnRender, &render>();

	application.run();
}

// Column count
// Normal flip