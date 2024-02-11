#include "transform.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/transform.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void updateModelMatrices(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };
	const auto view{ world.getView<Position, Rotation, Scale, LocalToWorld, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		const glm::vec3 position{ view.get<Position>(entity).position };
		const glm::quat rotation{ view.get<Rotation>(entity).rotation };
		const float scale{ view.get<Scale>(entity).scale };

		glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
		localToWorld = glm::scale(glm::mat4_cast(rotation), glm::vec3{ scale });
		reinterpret_cast<glm::vec3&>(localToWorld[3]) = position;
	}
}

}
