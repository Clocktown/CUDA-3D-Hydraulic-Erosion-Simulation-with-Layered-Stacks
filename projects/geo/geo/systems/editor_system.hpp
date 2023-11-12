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
	static void initializeDirectionalLight(Editor& editor);
	static void initializeCamera(Editor& editor);
	static void initializeTerrain(Editor& editor);
	static void initializeMaterial(Editor& editor);
	static void initializeRenderMesh(Editor& editor);
	static void initializeSimulation(Editor& editor);
	static void updateTerrain(Editor& editor);
	static void updateMaterial(Editor& editor);
	static void updateRenderMesh(Editor& editor);
	static void updateApplicationGUI(Editor& editor);
	static void updateCameraGUI(Editor& editor);
	static void updateTerrainGUI(Editor& editor);
	static void updateSimulationGUI(Editor& editor);
	static void updateRenderingGUI(Editor& editor);
};

}
