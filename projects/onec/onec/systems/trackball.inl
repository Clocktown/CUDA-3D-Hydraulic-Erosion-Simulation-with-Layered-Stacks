#include "trackball.hpp"
#include "../config/config.hpp"
#include "../core/window.hpp"
#include "../core/world.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include "../components/camera.hpp"
#include "../components/trackball.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void updateTrackballs(const entt::exclude_t<Excludes...> excludes)
{
	const Window& window{ getWindow() };
	const glm::vec2 mouseDelta{ window.getMouseDelta() };
	const float mouseScrollDelta{ window.getMouseScrollDelta().y };

	World& world{ getWorld() };
	const auto view{ world.getView<Transform, Trackball>(entt::exclude<Excludes...>) };

	for (const entt::entity entity : view)
	{
		ONEC_ASSERT(!world.hasComponent<Parent>(entity), "Entity must not have a parent component");

		Transform& transform{ view.get<Transform>(entity) };
		glm::vec3 position{ transform.position };
		glm::quat rotation{ transform.rotation };

		Trackball& trackball{ view.get<Trackball>(entity) };
		glm::vec3 target{ trackball.target };

		glm::mat3 localToWorld;
		glm::vec3 positionToTarget;
		float distance;

		if (glm::any(glm::epsilonNotEqual(position, target, glm::vec3{ glm::epsilon<float>() })))
		{
			positionToTarget = target - position;
			distance = glm::length(positionToTarget);

			localToWorld[2] = -positionToTarget / distance;
			localToWorld[0] = glm::normalize(glm::cross(glm::vec3{ 0.0f, 1.0f, 0.0f }, localToWorld[2]));
			localToWorld[1] = glm::cross(localToWorld[2], localToWorld[0]);

			rotation = glm::quat_cast(localToWorld);
		}
		else
		{
			localToWorld = glm::mat3_cast(rotation);
			positionToTarget = glm::vec3{ 0.0f };
			distance = 0.0f;
		}

		if (window.isKeyDown(trackball.panKey))
		{
			const glm::vec3 translation{ -trackball.panSensitivity * (mouseDelta.x * localToWorld[0] - mouseDelta.y * localToWorld[1]) };

			position += translation;
			target += translation;
		}

		const auto orbit{ [&]()
		{
			const float pitch{ glm::asin(positionToTarget.y / (distance + glm::epsilon<float>())) };
			float pitchDelta{ -trackball.orbitSensitivity * mouseDelta.y };

			pitchDelta = glm::clamp(pitch + pitchDelta, glm::radians(-89.5f), glm::radians(89.5f)) - pitch;

			const float yawDelta{ -trackball.orbitSensitivity * mouseDelta.x };

			rotation = glm::angleAxis(pitchDelta, localToWorld[0]) * rotation;
			rotation = glm::angleAxis(yawDelta, glm::vec3{ 0.0f, 1.0f, 0.0f }) * rotation;
			position = target + distance * (rotation * glm::vec3{ 0.0f, 0.0f, 1.0f });
		} };

		if (mouseScrollDelta != 0.0f)
		{
			const float maxTranslation{ distance - 0.5f };
			const glm::vec3 translation{ glm::min(trackball.zoomSensitivity * mouseScrollDelta, maxTranslation) * -localToWorld[2] };

			position += translation;

			if (window.isKeyDown(trackball.orbitKey))
			{
				positionToTarget = target - position;
				distance = glm::length(positionToTarget);

				orbit();
			}
		}
		else if (window.isKeyDown(trackball.orbitKey))
		{
			orbit();
		}

		transform.position = position;
		transform.rotation = rotation;
		trackball.target = target;
	}
}

}
