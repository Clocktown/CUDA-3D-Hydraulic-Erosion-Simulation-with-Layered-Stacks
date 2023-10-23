#include "geo/geo.hpp"
#include <onec/onec.hpp>

#define APPLICATION_NAME "Hydraulic Erosion"
#define WINDOW_SIZE glm::ivec2{ 1280, 720 }
#define MSAA 0
#define VSYNC_COUNT 0
#define TARGET_FRAME_RATE 0

void start()
{
	onec::Scene& scene{ onec::getScene() };

	const entt::entity camera{ scene.addEntity() };
	scene.addComponent<onec::PerspectiveCamera>(camera);
	scene.addComponent<onec::Trackball>(camera);
	scene.addComponent<onec::Position>(camera, glm::vec3{ 0.0f, 0.0f, 5.0f });
	scene.addComponent<onec::Rotation>(camera);
	scene.addComponent<onec::LocalToWorld>(camera);
	scene.addComponent<onec::WorldToView>(camera);
	scene.addComponent<onec::ViewToClip>(camera);

	scene.addSingleton<onec::ActiveCamera>(camera);

	const entt::entity directionalLight{ scene.addEntity() };
	scene.addComponent<onec::DirectionalLight>(directionalLight);
	scene.addComponent<onec::Rotation>(directionalLight, glm::quat{ glm::radians(glm::vec3{ -50.0f, 30.0f, 0.0f }) });
	scene.addComponent<onec::LocalToWorld>(directionalLight);
	scene.addComponent<onec::Static>(directionalLight);
}

int main()
{
	onec::Application& application{ onec::createApplication(APPLICATION_NAME, WINDOW_SIZE, MSAA) };
	application.setVSyncCount(VSYNC_COUNT);
	application.setTargetFrameRate(TARGET_FRAME_RATE);
	
	onec::Scene& scene{ onec::getScene() };

	scene.addSingleton<onec::Viewport>();
	scene.addSingleton<onec::Renderer>();
	scene.addSingleton<onec::MeshRenderer>();
	scene.addSingleton<onec::Lighting>();
	scene.addSingleton<onec::Screenshot>();
	scene.addSingleton<geo::TerrainRenderer>();
	scene.addSingleton<geo::GUI>();

	scene.addSystem<onec::OnStart, &start>();
	scene.addSystem<onec::OnStart, onec::ViewportSystem::update>();
	scene.addSystem<onec::OnStart, &onec::HierarchySystem::update>();
	scene.addSystem<onec::OnStart, &onec::TrackballSystem::update>();
	scene.addSystem<onec::OnStart, &onec::TransformSystem::start>();
	scene.addSystem<onec::OnStart, &onec::CameraSystem::start>();
	scene.addSystem<onec::OnStart, &onec::LightingSystem::start>();
	scene.addSystem<onec::OnStart, &onec::RenderSystem::start>();
	scene.addSystem<onec::OnStart, &onec::MeshRenderSystem::start>();
	scene.addSystem<onec::OnStart, &geo::TerrainRenderSystem::start>();
	scene.addSystem<onec::OnStart, &geo::GUISystem::start>();

	scene.addSystem<onec::OnUpdate, &onec::TitleBarSystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::ViewportSystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::HierarchySystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::TrackballSystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::TransformSystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::CameraSystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::RenderSystem::update>();
	scene.addSystem<onec::OnUpdate, &onec::ScreenshotSystem::update>();

	scene.addSystem<onec::OnPreRender, &onec::LightingSystem::update>();
	scene.addSystem<onec::OnRender, &onec::MeshRenderSystem::update>();
	scene.addSystem<onec::OnRender, &geo::TerrainRenderSystem::update>();
	scene.addSystem<onec::OnGUI, &geo::GUISystem::update>();
	
	application.run();
}
