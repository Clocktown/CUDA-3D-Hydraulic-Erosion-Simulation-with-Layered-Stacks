#pragma once

#include "../singletons/editor.hpp"
#include <onec/onec.hpp>

namespace geo
{

class EditorSystem
{
public:
	static void start();
	static void update();
private:
	static void addDirectionalLight();
	static void addCamera(Editor& editor);
	static void addTerrain(Editor& editor);
	static void addMaterial(Editor& editor);
	static void addRenderMesh(Editor& editor);
	static void addSimulation(Editor& editor);
	static void updateApplicationGUI(Editor& editor);
	static void updateCameraGUI(Editor& editor);
	static void updateTerrainGUI(Editor& editor);
	static void updateSimulationGUI(Editor& editor);
	static void updateRenderingGUI(Editor& editor);
};

}
