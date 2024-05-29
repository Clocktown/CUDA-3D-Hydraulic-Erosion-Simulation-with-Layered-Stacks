#include "geo/geo.hpp"
#include <onec/onec.hpp>

#include "geo/singletons/point_render_pipeline.hpp"

#include <fstream>
#include <filesystem>

#include <chrono>

void start()
{
	onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };

	world.addSingleton<onec::Viewport>();
	glm::vec3 backgroundColor = glm::vec3(0.3f, 0.65f, 0.98f);
	world.addSingleton<onec::AmbientLight>(backgroundColor, 37000.f);
	world.addSingleton<onec::RenderPipeline>();
	world.addSingleton<geo::PointRenderPipeline>();

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
		uniforms.useInterpolation = ui.rendering.useInterpolation;
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

		geo::PointRenderer& pointRenderer{ world.addComponent<geo::PointRenderer>(entity) };
		pointRenderer.material = material;
		pointRenderer.first = 0;
		pointRenderer.count = terrain.numValidColumns;

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

	if (ui.performance.measureRendering) {
		glFinish();
		timestamp = std::chrono::steady_clock::now();
	}
	onec::updateTrackballs();
	onec::updateModelMatrices(entt::exclude<onec::Static>);
	onec::updateViewMatrices(entt::exclude<onec::Static>);
	onec::updateProjectionMatrices();

	renderPipeline.update();
	renderPipeline.render();

	ui.performance.measureAll();

	if (ui.performance.measureRendering) {
		glFinish();
		auto time = std::chrono::steady_clock::now() - timestamp;
		float milliseconds = time.count() * 1e-6f;
		ui.performance.measurements["Rendering"].update(milliseconds);
	}
}

void render()
{
	onec::World& world{ onec::getWorld() };
	geo::PointRenderPipeline& pointRenderPipeline{ *world.getSingleton<geo::PointRenderPipeline>() };
	pointRenderPipeline.render();
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
}
