#include "geo/geo.hpp"
#include <onec/onec.hpp>

#define APPLICATION_NAME "Hydraulic Erosion"
#define WINDOW_SIZE glm::ivec2{ 1280, 720 }
#define MSAA 4

void start()
{
	onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };

	const entt::entity directionalLight{ world.addEntity() };
	world.addComponent<onec::DirectionalLight>(directionalLight);
	world.addComponent<onec::Rotation>(directionalLight, glm::quat{ glm::radians(glm::vec3{ -50.0f, 30.0f, 0.0f }) });
	world.addComponent<onec::LocalToWorld>(directionalLight);
	world.addComponent<onec::Static>(directionalLight);

	world.addSingleton<onec::Gravity>();
	world.addSingleton<onec::Viewport>();
	world.addSingleton<onec::Renderer>();
	world.addSingleton<onec::MeshRenderer>();
	world.addSingleton<onec::Lighting>();
	world.addSingleton<onec::Screenshot>();
	world.addSingleton<geo::Editor>();

	onec::RenderSystem::start();
	onec::MeshRenderSystem::start();
	onec::LightingSystem::start();
	geo::EditorSystem::start();

	onec::ViewportSystem::update();
	onec::HierarchySystem::update();
	onec::TrackballSystem::update();
	onec::TransformSystem::update();
	onec::CameraSystem::update();
}

void update()
{
	onec::TitleBarSystem::update();
	onec::ViewportSystem::update();
	geo::EditorSystem::update();
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
	onec::World& world{ onec::getWorld() };

	world.addSystem<onec::OnStart, &start>();
	world.addSystem<onec::OnUpdate, &update>();
	world.addSystem<onec::OnFixedUpdate, &fixedUpdate>();
	world.addSystem<onec::OnPreRender, &preRender>();
	world.addSystem<onec::OnRender, &render>();

	application.run();
}
