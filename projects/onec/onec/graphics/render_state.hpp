#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <limits>

namespace onec
{

class RenderState
{
public:
	void use() const;
	void disuse() const;

	void setColorMask(const glm::bvec4& colorMask);
	void setDepthFunction(const GLenum depthFuntion);
	void setDepthMask(const bool depthMask);
	void setStencilFunction(const GLenum stencilFunction);
	void setStencilReference(const int stencilReference);
	void setStencilMask(const unsigned int stencilMask);
	void setStencilOpPass(const GLenum stencilOpPass);
	void setStencilOpFail(const GLenum stencilOpFail);
	void setStencilOpDepthFail(const GLenum stencilOpDepthFail);
	void setBlendEquation(const GLenum blendEquation);
	void setBlendSrcFactor(const GLenum blendSrcFactor);
	void setBlendDstFactor(const GLenum blendDstFactor);
	void setBlendColor(const glm::vec4& blendColor);
	void setCullFace(const GLenum cullFace);
	void setPolygonMode(const GLenum polygonMode);
	void enableDepthTest();
	void enableStencilTest();
	void enableBlending();
	void enableCulling();
	void disableDepthTest();
	void disableStencilTest();
	void disableBlending();
	void disableCulling();

	const glm::bvec4& getColorMask() const;
	GLenum getDepthFunction() const;
	bool getDepthMask() const;
	GLenum getStencilFunction() const;
	int getStencilReference() const;
	unsigned int getStencilMask() const;
	GLenum getStencilOpPass() const;
	GLenum getStencilOpFail() const;
	GLenum getStencilOpDepthFail() const;
	GLenum getBlendEquation() const;
	GLenum getBlendSrcFactor() const;
	GLenum getBlendDstFactor() const;
	const glm::vec4& getBlendColor() const;
	GLenum getCullFace() const;
	GLenum getPolygonMode() const;
	bool isDepthTestEnabled() const;
	bool isStencilTestEnabled() const;
	bool isBlendingEnabled() const;
	bool isCullingEnabled() const;
private:
	glm::bvec4 m_colorMask{ true };
	GLenum m_depthFunction{ GL_LESS };
	bool m_depthMask{ true };
	GLenum m_stencilFunction{ GL_ALWAYS };
	int m_stencilReference{ 0 };
	unsigned int m_stencilMask{ std::numeric_limits<unsigned int>::max() };
	GLenum m_stencilOpPass{ GL_KEEP };
	GLenum m_stencilOpFail{ GL_KEEP };
	GLenum m_stencilOpDepthFail{ GL_KEEP };
	GLenum m_blendEquation{ GL_FUNC_ADD };
	GLenum m_blendSrcFactor{ GL_ONE };
	GLenum m_blendDstFactor{ GL_ZERO };
	glm::vec4 m_blendColor{ 0.0f };
	GLenum m_cullFace{ GL_BACK };
	GLenum m_polygonMode{ GL_FILL };
	bool m_isDepthTestEnabled{ true };
	bool m_isStencilTestEnabled{ false };
	bool m_isBlendingEnabled{ false };
	bool m_isCullingEnabled{ true };
};

}
