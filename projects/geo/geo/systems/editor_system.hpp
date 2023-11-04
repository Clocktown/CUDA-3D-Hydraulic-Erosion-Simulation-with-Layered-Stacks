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
	static void updateApplicationGUI(Editor& editor);
	static void updateCameraGUI(Editor& editor);
	static void updateTerrainGUI(Editor& editor);
	static void updateSimulationGUI(Editor& editor);
	static void updateRenderingGUI(Editor& editor);
};

}
