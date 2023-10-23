#pragma once

#include "material.hpp"
#include "texture.hpp"
#include "../device/phong_brdf.hpp"
#include <memory>

namespace onec
{

struct PhongBRDF : Material
{
	device::PhongBRDF uniforms;
	std::shared_ptr<Texture> diffuseColorMap{ nullptr };
	std::shared_ptr<Texture> specularColorMap{ nullptr };
	std::shared_ptr<Texture> emissionColorMap{ nullptr };
	std::shared_ptr<Texture> shininessMap{ nullptr };
	std::shared_ptr<Texture> normalMap{ nullptr };
};

}
