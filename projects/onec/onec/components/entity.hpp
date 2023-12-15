#pragma once

#include <cstddef>
#include <string>

namespace onec
{

struct Name
{
	std::string name;
};

template<std::size_t Index>
struct Layer
{

};

template<typename Type>
struct Disabled
{

};

}
