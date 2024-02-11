#ifndef ONEC_COLOR_GLSL
#define ONEC_COLOR_GLSL

vec3 adjustGamma(const vec3 color, const float gamma)
{
	return pow(color, vec3(gamma));
}

vec3 sRGBToLinear(const vec3 color)
{
	return adjustGamma(color, 2.2f);
}

vec3 linearToSRGB(const vec3 color)
{
	return adjustGamma(color, 1.0f / 2.2f);
}

vec3 applyReinhardToneMap(const vec3 luminance)
{
	return luminance / (luminance + 1.0f);
}

#endif
