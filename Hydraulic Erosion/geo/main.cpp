#include "geo/geo.hpp"
#include <onec/onec.hpp>

#define APPLICATION_NAME "Hydraulic Erosion"
#define WINDOW_SIZE glm::ivec2{ 1280, 720 }
#define MSAA 0
#define VSYNC_COUNT 0
#define TARGET_FRAME_RATE 0

void start()
{
	onec::Application& application{ onec::getApplication() };
	onec::Scene& scene{ onec::getScene() };

	const entt::entity camera{ scene.addEntity() };
	scene.addComponent<onec::PerspectiveCamera>(camera);
	scene.addComponent<onec::Trackball>(camera);
	scene.addComponent<onec::Position>(camera, glm::vec3{ 0.0f, 0.0f, 5.0f });
	scene.addComponent<onec::Rotation>(camera);
	scene.addComponent<onec::LocalToWorld>(camera);
	scene.addComponent<onec::WorldToView>(camera);
	scene.addComponent<onec::ViewToClip>(camera);

	const entt::entity directionalLight{ scene.addEntity() };
	scene.addComponent<onec::DirectionalLight>(directionalLight);
	scene.addComponent<onec::Rotation>(directionalLight, glm::quat{ glm::radians(glm::vec3{ -50.0f, 30.0f, 0.0f }) });
	scene.addComponent<onec::LocalToWorld>(directionalLight);
	scene.addComponent<onec::Static>(directionalLight);

	scene.addSingleton<onec::ActiveCamera>(camera);
	scene.addSingleton<onec::Gravity>();
	scene.addSingleton<onec::Viewport>();
	scene.addSingleton<onec::Renderer>();
	scene.addSingleton<onec::MeshRenderer>();
	scene.addSingleton<onec::Lighting>();
	scene.addSingleton<onec::Screenshot>();
	scene.addSingleton<geo::GUI>();

	onec::RenderSystem::start();
	onec::MeshRenderSystem::start();
	onec::LightingSystem::start();
	geo::GUISystem::start();

	onec::ViewportSystem::update();
	onec::HierarchySystem::update();
	onec::TrackballSystem::update();
	onec::TransformSystem::update();
	onec::CameraSystem::update();
}

void update()
{
	geo::GUISystem::update();
	onec::TitleBarSystem::update();
	onec::ViewportSystem::update();
	onec::HierarchySystem::update();
	onec::TrackballSystem::update();
	onec::TransformSystem::update(entt::exclude<onec::Static>);
	onec::CameraSystem::update(entt::exclude<onec::Static>);
	onec::RenderSystem::update();
	onec::ScreenshotSystem::update();
}

void fixedUpdate()
{

}

void preRender()
{
	onec::LightingSystem::update();
}

void render()
{
	onec::MeshRenderSystem::update();
}

int main()
{
	onec::Application& application{ onec::createApplication(APPLICATION_NAME, WINDOW_SIZE, MSAA) };
	application.setVSyncCount(VSYNC_COUNT);
	application.setTargetFrameRate(TARGET_FRAME_RATE);

	onec::Scene& scene{ onec::getScene() };
	scene.addSystem<onec::OnStart, &start>();
	scene.addSystem<onec::OnUpdate, &update>();
	scene.addSystem<onec::OnFixedUpdate, &fixedUpdate>();
	scene.addSystem<onec::OnPreRender, &preRender>();
	scene.addSystem<onec::OnRender, &render>();

	application.run();
}

// device namespace
// gl attach naming
