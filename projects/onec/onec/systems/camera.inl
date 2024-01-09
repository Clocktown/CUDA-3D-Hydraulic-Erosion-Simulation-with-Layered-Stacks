#include "camera.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
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
inline void updateViewMatrices(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };
	const auto& modelMatrices{ world.getStorage<LocalToWorld>() };

	{
		const auto view{ world.getView<Transform, ParentToView, Parent, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const Transform& transform{ view.get<Transform>(entity) };

			glm::mat4& parentToView{ view.get<ParentToView>(entity).parentToView };
			parentToView = glm::translate(glm::mat4_cast(glm::conjugate(transform.rotation)), -transform.position);
		}
	}

	{
		const auto view{ world.getView<Transform, WorldToView, Includes...>(entt::exclude<Parent, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const Transform& transform{ view.get<Transform>(entity) };

			glm::mat4& worldToView{ view.get<WorldToView>(entity).worldToView };
			worldToView = glm::translate(glm::mat4_cast(glm::conjugate(transform.rotation)), -transform.position);
		}
	}

	{
		const auto view{ world.getView<Parent, ParentToView, WorldToView, Includes...>(excludes) };
		
		for (const entt::entity entity : view)
		{
			const entt::entity parent{ view.get<Parent>(entity).parent };

			ONEC_ASSERT(modelMatrices.contains(parent), "Parent must have a local to world component");
			const glm::mat4 worldToParent{ glm::inverse(modelMatrices.get(parent).localToWorld) };
			const glm::mat4& parentToView{ view.get<ParentToView>(entity).parentToView };

			glm::mat4& worldToView{ view.get<WorldToView>(entity).worldToView };
			worldToView = parentToView * worldToParent;
		}
	}
}

template<typename... Includes, typename... Excludes>
inline void updateProjectionMatrices(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<Viewport>(), "World must have a viewport singleton");

	const Viewport viewport{ *world.getSingleton<Viewport>() };

	ONEC_ASSERT(viewport.offset.x >= 0, "Viewport offset x must be greater than or equal to 0");
	ONEC_ASSERT(viewport.offset.y >= 0, "Viewport offset y must be greater than or equal to 0");
	ONEC_ASSERT(viewport.size.x > 0, "Viewport size x must be greater than 0");
	ONEC_ASSERT(viewport.size.y > 0, "Viewport size y must be greater than 0");
	ONEC_ASSERT(viewport.aspect.x > 0.0f, "Viewport aspect x must be greater than 0");
	ONEC_ASSERT(viewport.aspect.y > 0.0f, "Viewport aspect y must be greater than 0");

	const float aspectRatio{ viewport.aspect.x * static_cast<float>(viewport.size.x) / (viewport.aspect.y * static_cast<float>(viewport.size.y)) };
	const glm::vec2 orthographicOffset{ viewport.offset };
	const glm::vec2 orthographicSize{ static_cast<float>(viewport.size.x) / viewport.aspect.y, static_cast<float>(viewport.size.y) / viewport.aspect.x };

	{
		const auto view{ world.getView<OrthographicCamera, ViewToClip, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const OrthographicCamera& orthographicCamera{ view.get<OrthographicCamera>(entity) };

			glm::mat4& viewToClip{ view.get<ViewToClip>(entity).viewToClip };
			viewToClip = glm::ortho(orthographicOffset.x, orthographicCamera.orthographicScale * orthographicSize.x, orthographicOffset.y, orthographicCamera.orthographicScale * orthographicSize.y, orthographicCamera.nearPlane, orthographicCamera.farPlane);
		}
	}

	{
		const auto view{ world.getView<PerspectiveCamera, ViewToClip>() };

		for (const entt::entity entity : view)
		{
			const PerspectiveCamera& perspectiveCamera{ view.get<PerspectiveCamera>(entity) };

			glm::mat4& viewToClip{ view.get<ViewToClip>(entity).viewToClip };
			viewToClip = glm::perspective(perspectiveCamera.fieldOfView, aspectRatio, perspectiveCamera.nearPlane, perspectiveCamera.farPlane);
		}
	}
}

}
