#include "render_state.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <limits>

namespace onec
{

void RenderState::use() const
{
	GL_CHECK_ERROR(glColorMask(m_colorMask.x, m_colorMask.y, m_colorMask.z, m_colorMask.w));

	if (m_isDepthTestEnabled)
	{
		GL_CHECK_ERROR(glEnable(GL_DEPTH_TEST));
		GL_CHECK_ERROR(glDepthMask(m_depthMask));
		GL_CHECK_ERROR(glDepthFunc(m_depthFunction));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
	}

	if (m_isStencilTestEnabled)
	{
		GL_CHECK_ERROR(glEnable(GL_STENCIL_TEST));
		GL_CHECK_ERROR(glStencilFunc(m_stencilFunction, m_stencilReference, m_stencilMask));
		GL_CHECK_ERROR(glStencilOp(m_stencilOpFail, m_stencilOpDepthFail, m_stencilOpPass));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_STENCIL_TEST));
	}

	if (m_isBlendingEnabled)
	{
		GL_CHECK_ERROR(glEnable(GL_BLEND));
		GL_CHECK_ERROR(glBlendEquation(m_blendEquation));
		GL_CHECK_ERROR(glBlendFunc(m_blendSrcFactor, m_blendDstFactor));
		GL_CHECK_ERROR(glBlendColor(m_blendColor.x, m_blendColor.y, m_blendColor.z, m_blendColor.w));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_BLEND));
	}

	if (m_isCullingEnabled)
	{
		GL_CHECK_ERROR(glEnable(GL_CULL_FACE));
		GL_CHECK_ERROR(glCullFace(m_cullFace));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
	}

	GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, m_polygonMode));
}

void RenderState::disuse() const
{
	GL_CHECK_ERROR(glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));

	GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
	GL_CHECK_ERROR(glDepthMask(GL_TRUE));
	GL_CHECK_ERROR(glDepthFunc(GL_LESS));

	GL_CHECK_ERROR(glDisable(GL_STENCIL_TEST));
	GL_CHECK_ERROR(glStencilFunc(GL_ALWAYS, 0, std::numeric_limits<unsigned int>::max()));
	GL_CHECK_ERROR(glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP));

	GL_CHECK_ERROR(glDisable(GL_BLEND));
	GL_CHECK_ERROR(glBlendEquation(GL_FUNC_ADD));
	GL_CHECK_ERROR(glBlendFunc(GL_ONE, GL_ZERO));
	GL_CHECK_ERROR(glBlendColor(0.0f, 0.0f, 0.0f, 0.0f));

	GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
	GL_CHECK_ERROR(glCullFace(GL_BACK));

	GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
}

void RenderState::setColorMask(const glm::bvec4& colorMask)
{
	m_colorMask = colorMask;
}

void RenderState::setDepthFunction(const GLenum depthFuntion)
{
	m_depthFunction = depthFuntion;
}

void RenderState::setDepthMask(const bool depthMask)
{
	m_depthMask = depthMask;
}

void RenderState::setStencilFunction(const GLenum stencilFunction)
{
	m_stencilFunction = stencilFunction;
}

void RenderState::setStencilReference(const int stencilReference)
{
	m_stencilReference = stencilReference;
}

void RenderState::setStencilMask(const unsigned int stencilMask)
{
	m_stencilMask = stencilMask;
}

void RenderState::setStencilOpPass(const GLenum stencilOpPass)
{
	m_stencilOpPass = stencilOpPass;
}

void RenderState::setStencilOpFail(const GLenum stencilOpFail)
{
	m_stencilOpFail = stencilOpFail;
}

void RenderState::setStencilOpDepthFail(const GLenum stencilOpDepthFail)
{
	m_stencilOpDepthFail = stencilOpDepthFail;
}

void RenderState::setBlendEquation(const GLenum blendEquation)
{
	m_blendEquation = blendEquation;
}

void RenderState::setBlendSrcFactor(const GLenum blendSrcFactor)
{
	m_blendSrcFactor = blendSrcFactor;
}

void RenderState::setBlendDstFactor(const GLenum blendDstFactor)
{
	m_blendDstFactor = blendDstFactor;
}

void RenderState::setBlendColor(const glm::vec4& blendColor)
{
	m_blendColor = blendColor;
}

void RenderState::setCullFace(const GLenum cullFace)
{
	m_cullFace = cullFace;
}

void RenderState::setPolygonMode(const GLenum polygonMode)
{
	m_polygonMode = polygonMode;
}

void RenderState::enableDepthTest()
{
	m_isDepthTestEnabled = true;
}

void RenderState::enableStencilTest()
{
	m_isStencilTestEnabled = true;
}

void RenderState::enableBlending()
{
	m_isBlendingEnabled = true;
}

void RenderState::enableCulling()
{
	m_isCullingEnabled = true;
}

void RenderState::disableDepthTest()
{
	m_isDepthTestEnabled = false;
}

void RenderState::disableStencilTest()
{
	m_isStencilTestEnabled = false;
}

void RenderState::disableBlending()
{
	m_isBlendingEnabled = false;
}

void RenderState::disableCulling()
{
	m_isCullingEnabled = false;
}

const glm::bvec4& RenderState::getColorMask() const
{
	return m_colorMask;
}

GLenum RenderState::getDepthFunction() const
{
	return m_depthFunction;
}

bool RenderState::getDepthMask() const
{
	return m_depthMask;
}

GLenum RenderState::getStencilFunction() const
{
	return m_stencilFunction;
}

int RenderState::getStencilReference() const
{
	return m_stencilReference;
}

unsigned int RenderState::getStencilMask() const
{
	return m_stencilMask;
}

GLenum RenderState::getStencilOpPass() const
{
	return m_stencilOpPass;
}

GLenum RenderState::getStencilOpFail() const
{
	return m_stencilOpFail;
}

GLenum RenderState::getStencilOpDepthFail() const
{
	return m_stencilOpDepthFail;
}

GLenum RenderState::getBlendEquation() const
{
	return m_blendEquation;
}

GLenum RenderState::getBlendSrcFactor() const
{
	return m_blendSrcFactor;
}

GLenum RenderState::getBlendDstFactor() const
{
	return m_blendDstFactor;
}

const glm::vec4& RenderState::getBlendColor() const
{
	return m_blendColor;
}

GLenum RenderState::getCullFace() const
{
	return m_cullFace;
}

GLenum RenderState::getPolygonMode() const
{
	return m_polygonMode;
}

bool RenderState::isDepthTestEnabled() const
{
	return m_isDepthTestEnabled;
}

bool RenderState::isStencilTestEnabled() const
{
	return m_isStencilTestEnabled;
}

bool RenderState::isBlendingEnabled() const
{
	return m_isBlendingEnabled;
}

bool RenderState::isCullingEnabled() const
{
	return m_isCullingEnabled;
}

}
