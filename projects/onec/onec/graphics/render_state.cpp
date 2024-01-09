#include "render_state.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <limits>

namespace onec
{

void RenderState::use()
{
	GL_CHECK_ERROR(glColorMask(colorMask.x, colorMask.y, colorMask.z, colorMask.w));

	if (depthTest)
	{
		GL_CHECK_ERROR(glEnable(GL_DEPTH_TEST));
		GL_CHECK_ERROR(glDepthMask(depthMask));
		GL_CHECK_ERROR(glDepthFunc(depthFunction));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
	}

	if (stencilTest)
	{
		GL_CHECK_ERROR(glEnable(GL_STENCIL_TEST));
		GL_CHECK_ERROR(glStencilFunc(stencilFunction, stencilReference, stencilMask));
		GL_CHECK_ERROR(glStencilOp(stencilFailOperation, stencilDepthFailOperation, stencilPassOperation));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_STENCIL_TEST));
	}

	if (blending)
	{
		GL_CHECK_ERROR(glEnable(GL_BLEND));
		GL_CHECK_ERROR(glBlendEquation(blendEquation));
		GL_CHECK_ERROR(glBlendFunc(blendSourceFactor, blendDestinationFactor));
		GL_CHECK_ERROR(glBlendColor(blendColor.x, blendColor.y, blendColor.z, blendColor.w));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_BLEND));
	}

	if (culling)
	{
		GL_CHECK_ERROR(glEnable(GL_CULL_FACE));
		GL_CHECK_ERROR(glCullFace(cullFace));
	}
	else
	{
		GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
	}

	GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, polygonMode));
}

void RenderState::disuse()
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

}
