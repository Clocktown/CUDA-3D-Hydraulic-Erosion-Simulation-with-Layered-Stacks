#include "transform.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>
#include <vector>

namespace onec
{

template<typename... Includes, typename... Excludes>
inline void updateModelMatrices(const entt::exclude_t<Excludes...> excludes)
{
	World& world{ getWorld() };

	{
		const auto view{ world.getView<Transform, LocalToParent, Parent, Includes...>(excludes) };

		for (const entt::entity entity : view)
		{
			const Transform& transform{ view.get<Transform>(entity) };

			glm::mat4& localToParent{ view.get<LocalToParent>(entity).localToParent };
			localToParent = glm::scale(glm::mat4_cast(transform.rotation), glm::vec3{ transform.scale });
			reinterpret_cast<glm::vec3&>(localToParent[3]) = transform.position;
		}
	}

	{
		const auto view{ world.getView<Transform, LocalToWorld, Includes...>(entt::exclude<Parent, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const Transform& transform{ view.get<Transform>(entity) };

			glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };
			localToWorld = glm::scale(glm::mat4_cast(transform.rotation), glm::vec3{ transform.scale });
			reinterpret_cast<glm::vec3&>(localToWorld[3]) = transform.position;
		}
	}

	{
		const auto view{ world.getView<Children, LocalToWorld, Includes...>(entt::exclude<Parent, Excludes...>) };

		for (const entt::entity entity : view)
		{
			const std::vector<entt::entity>& children{ view.get<Children>(entity).children };
			const glm::mat4& localToWorld{ view.get<LocalToWorld>(entity).localToWorld };

			struct
			{
				void operator()(const entt::entity entity, const glm::mat4& parentToWorld)
				{
					World& world{ getWorld() };

					if ((!world.hasComponent<Includes>(entity) || ...) || (world.hasComponent<Excludes>(entity) || ...))
					{
						return;
					}

					ONEC_ASSERT(world.hasComponent<LocalToParent>(entity), "Entity must have a local to parent component");
					ONEC_ASSERT(world.hasComponent<LocalToWorld>(entity), "Entity must have a local to world component");

					glm::mat4& localToWorld{ world.getComponent<LocalToWorld>(entity)->localToWorld };
					const glm::mat4& localToParent{ world.getComponent<LocalToParent>(entity)->localToParent };
					localToWorld = parentToWorld * localToParent;

					std::vector<entt::entity>* const children{ reinterpret_cast<std::vector<entt::entity>*>(world.getComponent<Children>(entity)) };

					if (children != nullptr)
					{
						for (const entt::entity child : *children)
						{
							operator()(child, localToWorld);
						}
					}
				}
			} propergateMatrix;

			for (const entt::entity child : children)
			{
				propergateMatrix(child, localToWorld);
			}
		}
	}
}

}
