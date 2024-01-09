#include "hierarchy.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include <entt/entt.hpp>
#include <vector>

namespace onec
{

void updateHierarchy()
{
	World& world{ getWorld() };
	auto& parents{ world.getStorage<Children>() };

	entt::dense_map<entt::entity, std::vector<entt::entity>> map;

	{
		const auto view{ world.getView<Parent, LocalToParent, LocalToWorld>() };

		for (const entt::entity entity : view)
		{
			const entt::entity parent{ view.get<Parent>(entity).parent };

			ONEC_ASSERT(world.hasEntity(parent), "Parent must refer to an existing entity");
			ONEC_ASSERT(world.hasComponent<LocalToWorld>(parent), "Parent must have local to world component");

            parents.emplace(parent);
	
			std::vector<entt::entity>& children{ map.try_emplace(parent).first->second };
			children.emplace_back(entity);
		}
	}

	{
		const auto view{ world.getView<Children>() };

		for (const entt::entity entity : view)
		{
			const auto iterator{ map.find(entity) };

			if (iterator != map.end())
			{
				view.get<Children>(entity).children = std::move(iterator->second);
			}
			else
			{
				parents.erase(entity);
			}
		}
	}
}

}
