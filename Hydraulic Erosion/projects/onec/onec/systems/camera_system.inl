#include "camera_system.hpp"
#include "../config/config.hpp"
#include "../core/scene.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include "../components/camera.hpp"
#include "../singletons/viewport.hpp"
#include "../singletons/camera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void CameraSystem::updateParentToView(const entt::exclude_t<Excludes...> excludes)
{
	updateMatrices<ParentToView, Parent, Includes...>(excludes);
}

template<typename... Includes, typename... Excludes>
inline void CameraSystem::updateWorldToView(const entt::exclude_t<Excludes...> excludes)
{
	Scene& scene{ getScene() };

	updateMatrices<WorldToView, Includes...>(entt::exclude<Parent, Excludes...>);

	const auto view{ scene.getView<Parent, ParentToView, WorldToView, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		const entt::entity parent{ view.get<Parent>(entity).parent };

		ONEC_ASSERT(scene.hasComponent<LocalToWorld>(parent), "Parent must have a local to world component");

		const glm::mat4 worldToParent{ glm::inverse(scene.getComponent<LocalToWorld>(parent)->localToWorld) };
		const glm::mat4& parentToView{ view.get<ParentToView>(entity).parentToView };

		glm::mat4& worldToView{ view.get<WorldToView>(entity).worldToView };
		worldToView = parentToView * worldToParent;
	}
}

template<typename... Includes, typename... Excludes>
inline void CameraSystem::updateViewToClip(const entt::exclude_t<Excludes...> excludes)
{
	Scene& scene{ getScene() };

	ONEC_ASSERT(scene.hasSingleton<Viewport>(), "Scene must have a viewport singleton");

	const Viewport viewport{ *scene.getSingleton<Viewport>() };

	ONEC_ASSERT(viewport.size.y != 0, "Viewport size y cannot be equal to 0");
	ONEC_ASSERT(viewport.aspect.x != 0.0f, "Viewport aspect x cannot be equal to 0");
	ONEC_ASSERT(viewport.aspect.y != 0.0f, "Viewport aspect y cannot be equal to 0");

	const float aspectRatio{ viewport.aspect.x * static_cast<float>(viewport.size.x) / (viewport.aspect.y * static_cast<float>(viewport.size.y)) };
	const glm::vec2 orthographicOffset{ viewport.offset };
	const glm::vec2 orthographicSize{ static_cast<float>(viewport.size.x) / viewport.aspect.y, static_cast<float>(viewport.size.y) / viewport.aspect.x };

	{
		const auto view{ scene.getView<OrthographicCamera, ViewToClip, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const OrthographicCamera& orthographicCamera{ view.get<OrthographicCamera>(entity) };

			glm::mat4& viewToClip{ view.get<ViewToClip>(entity).viewToClip };
			viewToClip = glm::ortho(orthographicOffset.x, orthographicCamera.scale * orthographicSize.x, orthographicOffset.y, orthographicCamera.scale * orthographicSize.y, orthographicCamera.nearPlane, orthographicCamera.farPlane);
		}
	}

	{
		const auto view{ scene.getView<PerspectiveCamera, ViewToClip>() };

		for (const entt::entity entity : view)
		{
			const PerspectiveCamera& perspectiveCamera{ view.get<PerspectiveCamera>(entity) };

			glm::mat4& viewToClip{ view.get<ViewToClip>(entity).viewToClip };
			viewToClip = glm::perspective(perspectiveCamera.fieldOfView, aspectRatio, perspectiveCamera.nearPlane, perspectiveCamera.farPlane);
		}
	}
}

template<typename Matrix, typename... Includes, typename... Excludes>
inline void CameraSystem::updateMatrices(const entt::exclude_t<Excludes...> excludes)
{
	Scene& scene{ getScene() };
	
	{
		const auto view{ scene.getView<Position, Matrix, Includes...>(entt::exclude<Rotation, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix[0] = glm::vec4{ 0.0f };
			matrix[1] = glm::vec4{ 0.0f };
			matrix[2] = glm::vec4{ 0.0f };
			matrix[3] = glm::vec4{ -position, 1.0f };
		}
	}

	{
		const auto view{ scene.getView<Rotation, Matrix, Includes...>(entt::exclude<Position, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::mat4_cast(glm::conjugate(rotation));
		}
	}
	
	{
		const auto view{ scene.getView<Position, Rotation, Matrix, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const glm::vec3& position{ view.get<Position>(entity).position };
			const glm::quat& rotation{ view.get<Rotation>(entity).rotation };

			glm::mat4& matrix{ reinterpret_cast<glm::mat4&>(view.get<Matrix>(entity)) };
			matrix = glm::translate(glm::mat4_cast(glm::conjugate(rotation)), -position);
		}
	}
}

}
