#include "geo/geo.hpp"
#include <onec/onec.hpp>

#include "geo/singletons/point_render_pipeline.hpp"
#include "geo/singletons/screen_texture_render_pipeline.hpp"

#include <fstream>
#include <filesystem>

#include <chrono>

GLuint oglQuery;

void start()
{
	onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };
	

	glGenQueries(1, &oglQuery);

	world.addSingleton<onec::Viewport>();
	glm::vec3 backgroundColor = glm::vec3(0.3f, 0.65f, 0.98f);
	world.addSingleton<onec::AmbientLight>(backgroundColor, 37000.f);
	world.addSingleton<onec::RenderPipeline>();
	world.addSingleton<geo::PointRenderPipeline>();
	world.addSingleton<geo::ScreenTextureRenderPipeline>();

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

	/* {
		const entt::entity entity{ world.addEntity() };

		world.addComponent<onec::SpotLight>(entity, onec::SpotLight{{0.f, 1.f, 0.f}, 10000000.0f, 200.f, glm::radians(45.0f), 0.15f });
		world.addComponent<onec::Position>(entity, glm::vec3(50.f, 30.f, 50.f));
		world.addComponent<onec::Rotation>(entity, glm::quat{ glm::radians(glm::vec3{ -90.0f, 0.0f, 60.0f }) });
		world.addComponent<onec::Scale>(entity);
		world.addComponent<onec::LocalToWorld>(entity);
		world.addComponent<onec::Static>(entity);
	}

		{
		const entt::entity entity{ world.addEntity() };

		world.addComponent<onec::PointLight>(entity, onec::PointLight{ {1.f, 0.f, 0.f}, 10000000.0f, 100.f });
		world.addComponent<onec::Position>(entity, glm::vec3(0.f, 30.f, 0.f));
		world.addComponent<onec::Rotation>(entity);
		world.addComponent<onec::Scale>(entity);
		world.addComponent<onec::LocalToWorld>(entity);
		world.addComponent<onec::Static>(entity);
	}

	{
		const entt::entity entity{ world.addEntity() };

		world.addComponent<onec::DirectionalLight>(entity, onec::DirectionalLight{ {0.5f,0.5f,0.75f}, 50000.f });
		world.addComponent<onec::Position>(entity);
		world.addComponent<onec::Rotation>(entity, glm::quat{ glm::radians(glm::vec3{ 50.0f, -30.0f, 0.0f }) });
		world.addComponent<onec::Scale>(entity);
		world.addComponent<onec::LocalToWorld>(entity);
		world.addComponent<onec::Static>(entity);
	}*/

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
		uniforms.useInterpolation = ui.rendering.useInterpolation;
		uniforms.renderSand = ui.rendering.renderSand;
		uniforms.renderWater = ui.rendering.renderWater;
		uniforms.gridSize = gridSize;
		uniforms.gridScale = gridScale;
		uniforms.maxLayerCount = terrain.maxLayerCount;
		uniforms.layerCounts = terrain.layerCountBuffer.getBindlessHandle();
		uniforms.heights = terrain.heightBuffer.getBindlessHandle();
		uniforms.stability = terrain.stabilityBuffer.getBindlessHandle();
		uniforms.indices = terrain.indicesBuffer.getBindlessHandle();

		const std::filesystem::path assets{ application.getDirectory() / "assets" };
		const auto material{ std::make_shared<onec::Material>() };
		const std::array<std::filesystem::path, 3> shaders{ assets / "shaders/point_terrain/simple_terrain.vert",
															assets / "shaders/point_terrain/simple_terrain.geom",
															assets / "shaders/point_terrain/simple_terrain.frag" };

		material->program = std::make_shared<onec::Program>(shaders);
		material->renderState = std::make_shared<onec::RenderState>();
		material->uniformBuffer.initialize(onec::asBytes(&uniforms, 1));

		const auto screenMaterial{ std::make_shared<onec::Material>() };
		const std::array<std::filesystem::path, 2> screenShaders{ assets / "shaders/screen_texture/screen_texture.vert",
															assets / "shaders/screen_texture/screen_texture.frag" };

		screenMaterial->program = std::make_shared<onec::Program>(screenShaders);
		screenMaterial->renderState = std::make_shared<onec::RenderState>();
		screenMaterial->uniformBuffer.initialize(onec::asBytes(&uniforms, 1));

		geo::PointRenderer& pointRenderer{ world.addComponent<geo::PointRenderer>(entity) };
		pointRenderer.material = material;
		pointRenderer.first = 0;
		pointRenderer.count = terrain.numValidColumns;
		pointRenderer.enabled = true;

		geo::ScreenTextureRenderer& screenTextureRenderer{ world.addComponent<geo::ScreenTextureRenderer>(entity) };
		screenTextureRenderer.material = screenMaterial;
		screenTextureRenderer.enabled = false;

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

	auto& ambientLight = *world.getSingleton<onec::AmbientLight>();
	auto& renderPipeline = *world.getSingleton<onec::RenderPipeline>();

	glm::vec3 clearColor = ambientLight.color * ambientLight.strength * (world.getComponent<onec::Exposure>(ui.camera.entity)->exposure);
	clearColor = clearColor / (clearColor + 1.0f);
	clearColor = onec::linearToSRGB(clearColor);
	renderPipeline.clearColor = glm::vec4{ clearColor, 1.0f };

	geo::updateTerrains();

	std::chrono::steady_clock::time_point timestamp;

	GLuint64 timeElapsed{ 0 };

	if (ui.rendering.renderScene && ui.performance.measureRendering) {
		glGetQueryObjectui64v(oglQuery, GL_QUERY_RESULT_NO_WAIT, &timeElapsed);
		if (timeElapsed > 0) {
			ui.performance.measurements["Rendering"].update(timeElapsed / 1000000.0);
		}
		glBeginQuery(GL_TIME_ELAPSED, oglQuery);
	}
	onec::updateTrackballs();
	onec::updateModelMatrices(entt::exclude<onec::Static>);
	onec::updateViewMatrices(entt::exclude<onec::Static>);
	onec::updateProjectionMatrices();

	if (ui.rendering.renderScene) {
		renderPipeline.update();
		renderPipeline.render();
	}

	ui.performance.measureAll();
	ui.performance.measurements["Frametime"].update(onec::getApplication().getUnscaledDeltaTime() * 1000.f);

	if (ui.rendering.renderScene && ui.performance.measureRendering) {
		glEndQuery(GL_TIME_ELAPSED);
	}
}

void render()
{
	onec::World& world{ onec::getWorld() };
	geo::PointRenderPipeline& pointRenderPipeline{ *world.getSingleton<geo::PointRenderPipeline>() };
	pointRenderPipeline.render();
	geo::ScreenTextureRenderPipeline& screenTextureRenderPipeline{ *world.getSingleton<geo::ScreenTextureRenderPipeline>() };
	screenTextureRenderPipeline.render();
}

int main()
{
	if (!std::filesystem::exists("cfg.txt")) {
		std::ofstream file("cfg.txt");
		file << 16;
	}
	int multisamples = 0;
	{
		std::ifstream file("cfg.txt");
		file >> multisamples;
		if (!file.good()) {
			multisamples = 0;
		}
	}

	onec::Application& application{ onec::createApplication("Hydraulic Erosion", glm::ivec2{ 1280, 720 }, multisamples) };
	application.setTargetFrameRate(3000);

	onec::Window& window{ onec::getWindow() };
	window.setSwapInterval(0);

	onec::World& world{ onec::getWorld() };
	world.addSystem<onec::OnStart, &start>();
	world.addSystem<onec::OnUpdate, &update>();
	world.addSystem<onec::OnRender, &render>();

	application.run();
	glDeleteQueries(1, &oglQuery);
}
