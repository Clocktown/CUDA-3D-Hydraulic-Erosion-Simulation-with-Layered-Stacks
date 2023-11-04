#include "hierarchy_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../components/hierarchy.hpp"
#include "../components/transform.hpp"
#include <entt/entt.hpp>
#include <vector>
#include <unordered_map>

namespace onec
{

void HierarchySystem::update()
{
	World& world{ getWorld() };
	std::unordered_map<entt::entity, std::vector<entt::entity>> map;

	{
		const auto view{ world.getView<Parent, LocalToParent, LocalToWorld>() };

		for (const entt::entity entity : view)
		{
			const entt::entity parent{ view.get<Parent>(entity).parent };

			ONEC_ASSERT(world.hasEntity(parent), "Parent must refer to an existing entity");
			ONEC_ASSERT(world.hasComponent<LocalToWorld>(parent), "Parent must have local to world component");

			world.addComponent<Children>(parent);

			std::vector<entt::entity>& children{ map.try_emplace(parent).first->second };
			children.emplace_back(entity);
		}
	}

	{
		const auto view{ world.getView<Children>() };

		for (const entt::entity entity : view)
		{
			const auto it{ map.find(entity) };

			if (it != map.end())
			{
				view.get<Children>(entity).children = std::move(it->second);
			}
			else
			{
				world.removeComponent<Children>(entity);
			}
		}
	}
}

}
