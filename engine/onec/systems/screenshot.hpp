#pragma once

#include <filesystem>
#include <string>

namespace onec
{

void takeScreenshot(std::string_view name = "Screenshot", const std::filesystem::path& folder = std::filesystem::current_path());

}