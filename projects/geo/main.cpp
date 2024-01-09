#include "geo/geo.hpp"
#include <onec/onec.hpp>

#define APPLICATION_NAME "Hydraulic Erosion"
#define WINDOW_SIZE glm::ivec2{ 1280, 720 }
#define MSAA_COUNT 4
#define TARGET_FRAMERATE 3000
#define SWAP_INTERVAL 0

#define GRID_SIZE glm::ivec3{ 256, 256, 8 }
#define GRID_SCALE 1.0f
#define VISUAL_SCALE 0.125f

entt::entity addCamera()
{
	onec::World& world{ onec::getWorld() };
	const entt::entity entity{ world.addEntity() };

	const glm::vec2 gridSize{ GRID_SIZE };
	const float maxLength{ VISUAL_SCALE * GRID_SCALE * glm::max(gridSize.x, gridSize.y) };

	world.addComponent<onec::PerspectiveCamera>(entity, glm::radians(60.0f), 0.1f, 1000.0f);
	world.addComponent<onec::Trackball>(entity);
	world.addComponent<onec::Transform>(entity, glm::vec3{ 0.0f, 0.25f * maxLength, 0.75f * maxLength }, 1.0f, glm::quat{ 1.0f, 0.0f, 0.0f, 0.0f });
	world.addComponent<onec::LocalToWorld>(entity);
	world.addComponent<onec::WorldToView>(entity);
	world.addComponent<onec::ViewToClip>(entity);

	world.setSingleton<onec::ActiveCamera>(entity);

	return entity;
}

entt::entity addDirectionalLight()
{
	onec::World& world{ onec::getWorld() };
	const entt::entity entity{ world.addEntity() };
	world.addComponent<onec::DirectionalLight>(entity).strength = 1.0f;
	world.addComponent<onec::Transform>(entity).rotation = glm::quat{ glm::radians(glm::vec3{ -50.0f, 30.0f, 0.0f }) };
	world.addComponent<onec::LocalToWorld>(entity);
	world.addComponent<onec::Static>(entity);
	
	return entity;
}

entt::entity addTerrain()
{
	const onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };

	const entt::entity entity{ world.addEntity() };
	const glm::ivec3 gridSize{ GRID_SIZE };

	world.addComponent<onec::Transform>(entity).scale = VISUAL_SCALE;
	world.addComponent<onec::LocalToWorld>(entity);

	const std::shared_ptr<geo::Terrain> terrain{ std::make_shared<geo::Terrain>(gridSize, GRID_SCALE) };
	const std::shared_ptr<geo::Material> material{ std::make_shared<geo::Material>(*terrain) };

	world.addComponent<geo::Simulation>(entity, terrain);
	world.addComponent<onec::Inactive<geo::Simulation>>(entity);

	onec::MeshRenderer& meshRenderer{ world.addComponent<onec::MeshRenderer>(entity) };
	meshRenderer.mesh = std::make_shared<onec::Mesh>(application.getDirectory() / "assets/meshes/cube.obj");
	meshRenderer.materials.emplace_back(material);
	meshRenderer.instanceCount = gridSize.x * gridSize.y * gridSize.z;

	return entity;
}

void start()
{
	CU_CHECK_ERROR(cudaSetDevice(0));

	onec::World& world{ onec::getWorld() };

	addDirectionalLight();
	geo::UI& ui{ world.addSingleton<geo::UI>() };
	ui.camera.entity = addCamera();
	ui.terrain.entity = addTerrain();
	ui.terrain.gridSize = GRID_SIZE;
	ui.terrain.gridScale = GRID_SCALE;

	onec::updateModelMatrices();

	world.addSingleton<onec::Viewport>();
	world.addSingleton<onec::AmbientLight>().strength = 1.0f;
	world.addSingleton<onec::RenderPipeline>().clearColor = glm::vec4{ 0.7f, 0.9f, 1.0f, 1.0f };
	world.addSingleton<onec::MeshRenderPipeline>();
	world.addSingleton<onec::RenderEnvironment>().update();
	world.addSingleton<onec::Gravity>();
}

void update()
{
	onec::Window& window{ onec::getWindow() };
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<onec::Viewport>(), "World must have a viewport singleton");
	ONEC_ASSERT(world.hasSingleton<onec::RenderPipeline>(), "World must have a render pipeline singleton");
	ONEC_ASSERT(world.hasSingleton<geo::UI>(), "World must have a render pipeline singleton");

	if (!window.isMinimized())
	{
		world.getSingleton<onec::Viewport>()->size = window.getFramebufferSize();
	}

	world.getSingleton<geo::UI>()->update();

	onec::updateTrackballs();
	onec::updateModelMatrices(entt::exclude<onec::Static>);
	onec::updateViewMatrices();
	onec::updateProjectionMatrices();

	world.getSingleton<onec::RenderPipeline>()->render();
}

void fixedUpdate(const onec::OnFixedUpdate event)
{
	geo::updateSimulation(event.fixedDeltaTime, entt::exclude<onec::Inactive<geo::Simulation>>);
}

void render()
{
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<onec::MeshRenderPipeline>(), "World must have a mesh render pipeline singleton");

	world.getSingleton<onec::MeshRenderPipeline>()->render(entt::exclude<onec::Inactive<onec::MeshRenderer>>);
}

int main()
{
	onec::Application& application{ onec::createApplication(APPLICATION_NAME, WINDOW_SIZE, MSAA_COUNT) };
	application.setTargetFrameRate(TARGET_FRAMERATE);

	onec::Window& window{ onec::getWindow() };
	window.setSwapInterval(SWAP_INTERVAL);

	onec::World& world{ onec::getWorld() };
	world.addSystem<onec::OnStart, &start>();
	world.addSystem<onec::OnUpdate, &update>();
	world.addSystem<onec::OnFixedUpdate, &fixedUpdate>();
	world.addSystem<onec::OnRender, &render>();

	application.run();
}
