#include "trackball_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../core/window.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include "../components/camera.hpp"
#include "../components/trackball.hpp"
#include "../singletons/camera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>

namespace onec
{

void updateTrackball()
{
	World& world{ getWorld() };
	ActiveCamera* const activeCamera{ world.getSingleton<ActiveCamera>() };

	if (activeCamera == nullptr)
	{
		return;
	}

	const entt::entity camera{ activeCamera->activeCamera };

	ONEC_ASSERT(!world.hasComponent<Parent>(camera), "Camera must not have a parent component");
	
	Trackball* const trackball{ world.getComponent<Trackball>(camera) };

	if (trackball == nullptr)
	{
		return;
	}

	ONEC_ASSERT(world.hasComponent<Position>(camera), "Camera must have a position component");
	ONEC_ASSERT(world.hasComponent<Rotation>(camera), "Camera must have a rotation component");

	glm::vec3& currentPosition{ world.getComponent<Position>(camera)->position };
	glm::quat& currentRotation{ world.getComponent<Rotation>(camera)->rotation };

	glm::vec3 target{ trackball->target };
	glm::vec3 position{ currentPosition };
	glm::quat rotation{ currentRotation };
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

	const Window& window{ getWindow() };
	const glm::vec2 mouseDelta{ window.getMouseDelta() };
	const float mouseScrollDelta{ window.getMouseScrollDelta().y };

	if (window.isKeyDown(trackball->panKey))
	{
		const glm::vec3 translation{ -trackball->panSensitivity * (mouseDelta.x * localToWorld[0] - mouseDelta.y * localToWorld[1]) };

		position += translation;
		target += translation;

		currentPosition = position;
		trackball->target = target;
	}

	if (mouseScrollDelta != 0.0f)
	{
		const float maxTranslation{ distance - 0.5f };
		const glm::vec3 translation{ glm::min(trackball->zoomSensitivity * mouseScrollDelta, maxTranslation) * -localToWorld[2] };

		position += translation;
		positionToTarget = target - position;
		distance = glm::length(positionToTarget);

		currentPosition = position;
	}

	if (window.isKeyDown(trackball->orbitKey))
	{
		const float pitch{ glm::asin(positionToTarget.y / (distance + glm::epsilon<float>())) };
		float pitchDelta{ -trackball->orbitSensitivity * mouseDelta.y };
		
		pitchDelta = glm::clamp(pitch + pitchDelta, glm::radians(-89.5f), glm::radians(89.5f)) - pitch;
		
		const float yawDelta{ -trackball->orbitSensitivity * mouseDelta.x };

		rotation = glm::angleAxis(pitchDelta, localToWorld[0]) * rotation;
		rotation = glm::angleAxis(yawDelta, glm::vec3{ 0.0f, 1.0f, 0.0f }) * rotation;
		position = target + distance * (rotation * glm::vec3{ 0.0f, 0.0f, 1.0f });

		currentPosition = position;
	}

	currentRotation = rotation;
}

}
