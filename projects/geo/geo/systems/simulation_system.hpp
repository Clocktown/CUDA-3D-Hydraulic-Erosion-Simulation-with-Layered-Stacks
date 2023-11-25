#pragma once

namespace geo
{

struct Simulation;

class SimulationSystem
{
public:
	static void initialize(Simulation& simulation);
	static void update();
	static void fixedUpdate();
private:
	static void simulate(Simulation& simulation, const float deltaTime);
	static void map(Simulation& simulation);
	static void unmap(Simulation& simulation);
};

}
