#pragma once

#include <filesystem>
#include <string>

namespace onec
{

struct Name
{
	std::string name;
};

struct File
{
	std::filesystem::path file;
};

}
