#include "camera.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include "../components/camera.hpp"
#include "../singletons/viewport.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void updateViewMatrices(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };
	const auto view{ world.getView<Position, Rotation, WorldToView, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		const glm::vec3 position{ view.get<Position>(entity).position };
		const glm::quat rotation{ view.get<Rotation>(entity).rotation };

		glm::mat4& worldToView{ view.get<WorldToView>(entity).worldToView };
		worldToView = glm::translate(glm::mat4_cast(glm::conjugate(rotation)), -position);
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

	const glm::vec2 offset{ viewport.offset };
	const glm::vec2 size{ viewport.size };
	const float aspectRatio{ size.x / size.y };

	{
		const auto view{ world.getView<OrthographicCamera, ViewToClip, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const OrthographicCamera orthographicCamera{ view.get<OrthographicCamera>(entity) };
	
			glm::mat4& viewToClip{ view.get<ViewToClip>(entity).viewToClip };
			viewToClip = glm::ortho(offset.x, orthographicCamera.orthographicScale * size.x, offset.y, orthographicCamera.orthographicScale * size.y, orthographicCamera.nearPlane, orthographicCamera.farPlane);
		}
	}

	{
		const auto view{ world.getView<PerspectiveCamera, ViewToClip>(excludes) };

		for (const entt::entity entity : view)
		{
			const PerspectiveCamera perspectiveCamera{ view.get<PerspectiveCamera>(entity) };

			glm::mat4& viewToClip{ view.get<ViewToClip>(entity).viewToClip };
			viewToClip = glm::perspective(perspectiveCamera.fieldOfView, aspectRatio, perspectiveCamera.nearPlane, perspectiveCamera.farPlane);
		}
	}
}

}
