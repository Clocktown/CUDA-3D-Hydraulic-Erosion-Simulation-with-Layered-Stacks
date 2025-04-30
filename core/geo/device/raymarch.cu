#include "simulation.hpp"

#include <numbers>

namespace geo {
	namespace device {

constexpr int MAX_QUADTREE_LAYER = geo::NUM_QUADTREE_LAYERS - 1;
constexpr int AIR = CEILING;

struct Ray {
	glm::vec2 t;
	glm::vec3 o;
	glm::vec3 dir;
	glm::vec3 i_dir;
	glm::bvec2 orientation;
};

struct BoxHit {
	float t{ FLT_MAX };
	bool hit{ false };
	int material{ AIR };
	glm::vec2 boxHeights{ 0.f };
	glm::vec3 normal{ 0.f };
	glm::vec3 pos{ 0.f };
};

struct DDAExitLevelState {
	glm::ivec2 currentCell;
	glm::ivec2 exit;
	glm::ivec2 step;

	glm::vec2 deltaT;
	glm::vec2 T;

	glm::vec2 bmin2D;
	glm::vec2 bmax2D;

	float exitLevels[geo::NUM_QUADTREE_LAYERS];
};

struct DDAMissState {
	glm::ivec2 currentCell;
	glm::ivec2 exit;
	glm::ivec2 step;

	glm::vec2 deltaT;
	glm::vec2 T;

	glm::vec2 bmin2D;
	glm::vec2 bmax2D;

	int miss;
};

struct DDAState {
	glm::ivec2 currentCell;
	glm::ivec2 exit;
	glm::ivec2 step;

	glm::vec2 deltaT;
	glm::vec2 T;

	glm::vec2 bmin2D;
	glm::vec2 bmax2D;
};

struct PhongBRDF
{
	glm::vec3 diffuseReflectance;
	glm::vec3 specularReflectance;
	float shininess;
	float alpha;
	glm::vec3 position;
	glm::vec3 normal;
	float F;
};

constexpr float epsilon = 1.192092896e-07f;
constexpr float bigEpsilon = 1e-4f;
constexpr float rPi = 0.318309886f;


__device__ __forceinline__ glm::vec3 adjustGamma(const glm::vec3 color, const float gamma)
{
	return pow(color, glm::vec3(gamma));
}

__device__ __forceinline__ glm::vec3 sRGBToLinear(const glm::vec3 color)
{
	return adjustGamma(color, 2.2f);
}

__device__ __forceinline__ glm::vec3 linearToSRGB(const glm::vec3 color)
{
	return adjustGamma(color, 1.0f / 2.2f);
}

__device__ __forceinline__ glm::vec3 applyReinhardToneMap(const glm::vec3 luminance)
{
	return luminance / (luminance + 1.0f);
}

__device__ __forceinline__ float getAttenuation(const float distance, const float range)
{
	float ratio = distance / range;
	return 1.f / (ratio * ratio + 1.0f);
}

__device__ __forceinline__ glm::vec3 getPointLightIlluminance(const onec::RenderPipelineUniforms::PointLight pointLight, const glm::vec3 direction, const float distance, const glm::vec3 normal)
{
	return getAttenuation(distance, pointLight.range) * pointLight.intensity * glm::max(glm::dot(-direction, normal), 0.0f);
}

__device__ __forceinline__ glm::vec3 getSpotLightIlluminance(const onec::RenderPipelineUniforms::SpotLight spotLight, const glm::vec3 direction, const float distance, const glm::vec3 normal)
{
	const float cutOff = glm::dot(spotLight.direction, direction);

	if (cutOff < spotLight.outerCutOff)
	{
		return glm::vec3(0.0f);
	}

	const float attenuation = glm::min((cutOff - spotLight.outerCutOff) / (spotLight.innerCutOff - spotLight.outerCutOff + epsilon), 1.0f) * getAttenuation(distance, spotLight.range);

	return attenuation * spotLight.intensity * glm::max(glm::dot(-direction, normal), 0.0f);
}

__device__ __forceinline__ glm::vec3 getDirectionalLightIlluminance(const onec::RenderPipelineUniforms::DirectionalLight directionalLight, const glm::vec3 direction, const glm::vec3 normal)
{
	return directionalLight.luminance * glm::max(glm::dot(-direction, normal), 0.0f);
}

__device__ __forceinline__ glm::vec3 evaluatePhongBRDF(const PhongBRDF phongBRDF, const glm::vec3 lightDirection, const glm::vec3 viewDirection)
{
	const glm::vec3 reflection = reflect(-lightDirection, phongBRDF.normal);
	const float cosPhi = glm::max(dot(viewDirection, reflection), 0.0f);

	const glm::vec3 diffuseBRDF = rPi * phongBRDF.diffuseReflectance;
	const glm::vec3 specularBRDF = 0.5f * rPi * (phongBRDF.shininess + 2.0f) * pow(cosPhi, phongBRDF.shininess) * phongBRDF.specularReflectance;

	return diffuseBRDF + specularBRDF;
}

__device__ __forceinline__ glm::vec3 getAmbientLightLuminance(const onec::RenderPipelineUniforms::AmbientLight ambientLight, const PhongBRDF phongBRDF)
{
	return (rPi * phongBRDF.diffuseReflectance) * ambientLight.luminance;
}

__device__ __forceinline__ glm::vec3 getPointLightLuminance(const onec::RenderPipelineUniforms::PointLight pointLight, const PhongBRDF phongBRDF, const glm::vec3 direction)
{
	const glm::vec3 lightVector = pointLight.position - phongBRDF.position;
	const float lightDistance = length(lightVector);
	const glm::vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const glm::vec3 luminance = evaluatePhongBRDF(phongBRDF, lightDirection, direction) *
		                   getPointLightIlluminance(pointLight, -lightDirection, lightDistance, phongBRDF.normal);

	return luminance;
}

__device__ __forceinline__ glm::vec3 getSpotLightLuminance(const onec::RenderPipelineUniforms::SpotLight spotLight, const PhongBRDF phongBRDF, const glm::vec3 direction)
{
	const glm::vec3 lightVector = spotLight.position - phongBRDF.position;
	const float lightDistance = length(lightVector);
	const glm::vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const glm::vec3 luminance = evaluatePhongBRDF(phongBRDF, lightDirection, direction) *
		                   getSpotLightIlluminance(spotLight, -lightDirection, lightDistance, phongBRDF.normal);

	return luminance;
}

__device__ __forceinline__ glm::vec3 getDirectionalLightLuminance(const onec::RenderPipelineUniforms::DirectionalLight directionalLight, const PhongBRDF phongBRDF, const glm::vec3 direction)
{
	const glm::vec3 lightDirection = -directionalLight.direction;
	const glm::vec3 luminance = evaluatePhongBRDF(phongBRDF, lightDirection, direction) *
		                   getDirectionalLightIlluminance(directionalLight, directionalLight.direction, phongBRDF.normal);

	return luminance;
}

__device__ __forceinline__ bool intersection(const glm::vec3& b_min, const glm::vec3& b_max, const glm::vec3& r_o, const glm::vec3& inv_r_dir, glm::vec2& t) {
    glm::vec3 t1 = (b_min - r_o) * inv_r_dir;
    glm::vec3 t2 = (b_max - r_o) * inv_r_dir;

    glm::vec3 vtmin = glm::min(t1, t2);
    glm::vec3 vtmax = glm::max(t1, t2);

    t.x = glm::max(vtmin.x, glm::max(vtmin.y, vtmin.z));
    t.y = glm::min(vtmax.x, glm::min(vtmax.y, vtmax.z));

	return t.y >= t.x && t.y >= 0;
}

__device__ __forceinline__ float intersectionVolume(const glm::vec3& b_min0, const glm::vec3& b_max0, const glm::vec3& b_min1, const glm::vec3& b_max1) {
	const glm::vec3 b_min = glm::max(b_min0, b_min1);
	const glm::vec3 b_max = glm::min(b_max0, b_max1);
	const glm::vec3 d = glm::max(b_max - b_min, glm::vec3(0));

	return d.x * d.y * d.z;
}

template <class State>
__device__ __forceinline__ void updateStateAABB(State& state, float gridScale) {
	state.bmin2D = glm::vec2(state.currentCell) * gridScale;
	state.bmax2D = state.bmin2D + gridScale;
}

template <class State>
__device__ __forceinline__ void updateStateAABB(State& state, int currentLevel) {
	return updateStateAABB(state, currentLevel < 0 ? simulation.gridScale : simulation.quadTree[currentLevel].gridScale);
}

template <class State>
__device__ __forceinline__ void calculateDDAState(State& state, const Ray& ray, const glm::ivec2& gridSize, float gridScale, float rGridScale) {
	const glm::vec3 rayStart = ray.o + ray.t.x * ray.dir;

	state.currentCell = glm::clamp(
		glm::ivec2(
			glm::floor(glm::vec2(rayStart.x, rayStart.z) * rGridScale)),
		glm::ivec2(0),
		gridSize - glm::ivec2(1)
	);
	state.exit = glm::ivec2(
		ray.orientation.x ? -1 : gridSize.x,
		ray.orientation.y ? -1 : gridSize.y
	);
	state.step = glm::ivec2(
		ray.orientation.x ? -1 : 1,
		ray.orientation.y ? -1 : 1
	);

	state.deltaT = glm::abs(glm::vec2(
		gridScale * ray.i_dir.x,
		gridScale * ray.i_dir.z)
	);
	state.T = ray.t.x + ((glm::vec2(state.currentCell) + (1.f - glm::vec2(ray.orientation))) * gridScale - glm::vec2(rayStart.x, rayStart.z)) * glm::vec2(ray.i_dir.x, ray.i_dir.z);

	updateStateAABB(state, gridScale);

	if constexpr (std::is_same_v<State, DDAMissState>) {
		state.miss = 0;
	}
}


template <class State>
__device__ __forceinline__ void calculateDDAState(State& state, const Ray& ray, int currentLevel) {
	if (currentLevel < 0) {
		return calculateDDAState(state, ray, simulation.gridSize, simulation.gridScale, simulation.rGridScale);
	}
	else {
		return calculateDDAState(state, ray, simulation.quadTree[currentLevel].gridSize, simulation.quadTree[currentLevel].gridScale, simulation.quadTree[currentLevel].rGridScale);
	}
}


template <class State>
__device__ __forceinline__ void calculateDDAState(State& state, const Ray& ray) {
	return calculateDDAState(state, ray, simulation.gridSize, simulation.gridScale, simulation.rGridScale);
}

__device__ __forceinline__ Ray createRay(const glm::vec3& o, const glm::vec3& dir) {
	Ray ray;
	ray.o = o;
	ray.dir = dir;
	ray.orientation = glm::bvec2(ray.dir.x < 0.f, ray.dir.z < 0.f);

	const glm::vec3 inv_r_dir = 1.0f / ray.dir;
	const auto inf_mask = glm::isinf(inv_r_dir);
	ray.i_dir = glm::vec3(
		inf_mask.x ? 0.f : inv_r_dir.x,
		inf_mask.y ? 0.f : inv_r_dir.y,
		inf_mask.z ? 0.f : inv_r_dir.z
	);
	ray.t.x = 0.f;
	ray.t.y = FLT_MAX;
	return ray;
}

__device__ __forceinline__ Ray createRay(const glm::ivec2& index) {
	const glm::vec3 o = simulation.rendering.i_scale * simulation.rendering.camPos;
	const glm::vec3 pW = simulation.rendering.i_scale * (simulation.rendering.lowerLeft + (index.x + 0.5f) * simulation.rendering.rightVec + (index.y + 0.5f) * simulation.rendering.upVec);

	return createRay(
		o,
		glm::normalize(pW - o)
	);
}

template <class State>
__device__ __forceinline__ void intersectColumnsAny(const State& state, const Ray& ray, BoxHit& hit) {
	int flatIndex{ flattenIndex(state.currentCell, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	float floor = -FLT_MAX;

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
		terrainHeights[SAND] += terrainHeights[BEDROCK];
		terrainHeights[WATER] += terrainHeights[SAND];

		glm::vec2 tempT;
		bool intersect = intersection(glm::vec3(state.bmin2D.x, floor, state.bmin2D.y), glm::vec3(state.bmax2D.x, terrainHeights[WATER], state.bmax2D.y), ray.o, ray.i_dir, tempT);

		if (intersect && tempT.x < ray.t.y) {
			hit.hit = true;
			break;
		}

		floor = terrainHeights[CEILING];
	}
}

template <class State>
__device__ __forceinline__ void intersectColumns(const State& state, const Ray& ray, BoxHit& hit) {
	int flatIndex{ flattenIndex(state.currentCell, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	float floor = -FLT_MAX;

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
		terrainHeights[SAND] += terrainHeights[BEDROCK];
		terrainHeights[WATER] += terrainHeights[SAND];

		glm::vec2 tempT;
		bool intersect = intersection(glm::vec3(state.bmin2D.x, floor, state.bmin2D.y), glm::vec3(state.bmax2D.x, terrainHeights[WATER], state.bmax2D.y), ray.o, ray.i_dir, tempT);

		if (intersect && tempT.x < hit.t) {
			hit.hit = true;
			hit.boxHeights = glm::vec2(floor, terrainHeights[WATER]);
			hit.t = tempT.x;
			const float height = ray.o.y + hit.t * ray.dir.y;
			hit.material = WATER;
			for (int i = SAND; i >= BEDROCK; --i) {
				if (height <= terrainHeights[i] + bigEpsilon) hit.material = i;
			}
		}

		floor = terrainHeights[CEILING];
	}
}

template <class State>
__device__ __forceinline__ void intersectQuadTreeColumns(const State& state, const Ray& ray, BoxHit& hit, int currentLevel) {
	int flatIndex{ flattenIndex(state.currentCell, simulation.quadTree[currentLevel].gridSize) };
	const int layerCount{ simulation.quadTree[currentLevel].layerCounts[flatIndex] };

	float floor = -FLT_MAX;

	const glm::vec2 bmin2D = glm::vec2(state.currentCell) * simulation.quadTree[currentLevel].gridScale;
	const glm::vec2 bmax2D = bmin2D + simulation.quadTree[currentLevel].gridScale;

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.quadTree[currentLevel].layerStride)
	{
		const auto terrainHeights = glm::cuda_cast(simulation.quadTree[currentLevel].heights[flatIndex]);
		const float totalTerrainHeight = terrainHeights[QFULLHEIGHT];

		glm::vec2 tempT;
		bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), ray.o, ray.i_dir, tempT);
		if (intersect && (tempT.y > ray.t.x)) {
			hit.hit = true;
			hit.boxHeights = glm::vec2(floor, totalTerrainHeight);
			hit.t = glm::max(tempT.x, ray.t.x);
			break;
		}

		floor = terrainHeights[QCEILING];
	}
}

template <class State>
__device__ __forceinline__ void intersectSceneAsBoxes(const State& state, const Ray& ray, BoxHit& hit, int currentLevel) {
	if (currentLevel < 0) {
		// Terrain column intersection
		intersectColumns(state, ray, hit);
	}
	else {
		// QuadTree Column Intersection
		intersectQuadTreeColumns(state, ray, hit, currentLevel);
	}
}

template <class State>
__device__ __forceinline__ void intersectSceneAsBoxesAny(const State& state, const Ray& ray, BoxHit& hit, int currentLevel) {
	if (currentLevel < 0) {
		// Terrain column intersection
		intersectColumnsAny(state, ray, hit);
	}
	else {
		// QuadTree Column Intersection
		intersectQuadTreeColumns(state, ray, hit, currentLevel);
	}
}

template <class State>
__device__ __forceinline__ bool advanceDDA(State& state, Ray& ray, int currentLevel) {
	if constexpr (std::is_same_v<State, DDAMissState>) {
		state.miss++;
	}
	// Check step axis
	const bool axis = state.T.x >= state.T.y;
	// Shorten Ray
	ray.t.x = state.T[axis];
	if (ray.t.y < ray.t.x) return true;
	// Advance
	state.currentCell[axis] += state.step[axis];
	if (state.currentCell[axis] == state.exit[axis]) return true;
	state.T[axis] += state.deltaT[axis];

	
	updateStateAABB(state, currentLevel);
	return false;
}

template <class State>
__device__ __forceinline__ void resolveBoxHit(BoxHit& hit, const State& state, const Ray& ray) {
	hit.pos = ray.o + hit.t * ray.dir;
	glm::vec3 bmin = glm::vec3(state.bmin2D.x, hit.boxHeights.x, state.bmin2D.y);
	glm::vec3 bmax = glm::vec3(state.bmax2D.x, hit.boxHeights.y, state.bmax2D.y);
	glm::vec3 diffMin = glm::abs(hit.pos - bmin);
	glm::vec3 diffMax = glm::abs(hit.pos - bmax);
	glm::vec3 distances = glm::min(glm::abs(diffMin), glm::abs(diffMax));

	// Determine the closest face based on the minimum distance
	if (distances.x <= distances.y && distances.x <= distances.z) {
		hit.normal.x = (diffMin.x < diffMax.x) ? -1.0f : 1.0f;
	} else if (distances.y <= distances.x && distances.y <= distances.z) {
		hit.normal.y = (diffMin.y < diffMax.y) ? -1.0f : 1.0f;
	} else {
		hit.normal.z = (diffMin.z < diffMax.z) ? -1.0f : 1.0f;
	}
}

template <class State, bool Shadow>
__device__ __forceinline__ BoxHit traceRayBoxes(Ray& ray) {
	BoxHit hit;
	ray.o += simulation.rendering.gridOffset;
	const glm::vec3 bmin = glm::vec3(0.f, -FLT_MAX, 0.f);
	const glm::vec3 bmax = glm::vec3(2.f * simulation.rendering.gridOffset.x, FLT_MAX, 2.f * simulation.rendering.gridOffset.z);

	glm::vec2 t;
	bool intersect = intersection(bmin, bmax, ray.o, ray.i_dir, t);
	ray.t.x = glm::max(ray.t.x, t.x);
	ray.t.y = glm::min(ray.t.y, t.y);
	if (ray.t.y <= ray.t.x) intersect = false;

	int currentLevel = MAX_QUADTREE_LAYER;

	int steps = 0;
	if (intersect) {
		// DDA prep
		State state;
		calculateDDAState(state, ray, currentLevel);
		// DDA
		while (true) {
			steps++;

			// Simpler alternative to backtracking, no stack in state needed, has better performance
			if constexpr (std::is_same_v<State, DDAMissState>) {
				if ((currentLevel < MAX_QUADTREE_LAYER) && state.miss == simulation.rendering.missCount) {
					currentLevel++;
					calculateDDAState(state, ray, currentLevel);
					continue;
				}
			}

			hit.t = FLT_MAX;
			if constexpr (Shadow) {
				intersectSceneAsBoxesAny(state, ray, hit, currentLevel);
			}
			else {
				intersectSceneAsBoxes(state, ray, hit, currentLevel);
			}

			if (hit.hit && (currentLevel < 0)) {
				// Hit actual terrain geometry
				if constexpr (!Shadow) resolveBoxHit(hit, state, ray);
				break;
			}
			else if (hit.hit) {
				// Remember Exit for Backtracking
				if constexpr (std::is_same_v<State, DDAExitLevelState>) {
					const bool axis = state.T.x >= state.T.y;
					state.exitLevels[currentLevel] = state.T[axis];
				}
				// Hit coarse geometry in QuadTree. Move to finer level.
				ray.t.x = glm::max(ray.t.x, hit.t);
				currentLevel--;
				calculateDDAState(state, ray, currentLevel);
				hit.hit = false;
				continue;
			}

			// Backtracking - very delicate, easy to implement wrong
			if constexpr (std::is_same_v<State, DDAExitLevelState>) {
				bool movedUp = false;
				while ((currentLevel < MAX_QUADTREE_LAYER) && (ray.t.x >= state.exitLevels[currentLevel + 1] - bigEpsilon)) {
					currentLevel++;
					movedUp = true;
				}
				if (movedUp) {
					const float oldRay = ray.t.x;
					ray.t.x = glm::max(ray.t.x + bigEpsilon, state.exitLevels[currentLevel] + bigEpsilon);
					if (ray.t.y < ray.t.x) break;
					calculateDDAState(state, ray, currentLevel);
					ray.t.x = oldRay;
					continue;
				}
			}
				
			if (advanceDDA(state, ray, currentLevel)) break;
		}
	}
	ray.o -= simulation.rendering.gridOffset;
	hit.pos -= simulation.rendering.gridOffset;

	return hit;
}

template <class State>
__device__ __forceinline__ glm::vec3 shadePhongBRDFBoxes(const PhongBRDF phongBRDF)
{
	const glm::vec3 viewDirection = glm::normalize(simulation.rendering.camPos - phongBRDF.position);
	glm::vec3 luminance = getAmbientLightLuminance(simulation.rendering.uniforms->ambientLight, phongBRDF);

	for (int i = 0; i < simulation.rendering.uniforms->pointLightCount; ++i)
	{
		const auto lum = getPointLightLuminance(simulation.rendering.uniforms->pointLights[i], phongBRDF, viewDirection);
		if (lum != glm::vec3(0.f)) {
			glm::vec3 direction = simulation.rendering.uniforms->pointLights[i].position - phongBRDF.position;
			float distance = glm::length(direction);
			direction = direction / distance;
			Ray ray{ createRay(phongBRDF.position + bigEpsilon * direction, direction) };
			ray.t.y = distance - bigEpsilon;
			auto hit = traceRayBoxes<State, true>(ray);
			if(!hit.hit) luminance += lum;
		}
	}

	for (int i = 0; i < simulation.rendering.uniforms->spotLightCount; ++i)
	{
		const auto lum = getSpotLightLuminance(simulation.rendering.uniforms->spotLights[i], phongBRDF, viewDirection);
		if (lum != glm::vec3(0.f)) {
			glm::vec3 direction = simulation.rendering.uniforms->spotLights[i].position - phongBRDF.position;
			float distance = glm::length(direction);
			direction = direction / distance;
			Ray ray{ createRay(phongBRDF.position + bigEpsilon * direction, direction) };
			ray.t.y = distance - bigEpsilon;
			auto hit = traceRayBoxes<State, true>(ray);
			if(!hit.hit) luminance += lum;
		}
	}

	for (int i = 0; i < simulation.rendering.uniforms->directionalLightCount; ++i)
	{
		const auto lum = getDirectionalLightLuminance(simulation.rendering.uniforms->directionalLights[i], phongBRDF, viewDirection);
		if (lum != glm::vec3(0.f)) {
			Ray ray{ createRay(phongBRDF.position - bigEpsilon * simulation.rendering.uniforms->directionalLights[i].direction, -simulation.rendering.uniforms->directionalLights[i].direction) };
			auto hit = traceRayBoxes<State, true>(ray);
			if(!hit.hit) luminance += lum;
		}
	}

	return simulation.rendering.uniforms->exposure * luminance;
}

template <class State>
__global__ void raymarchDDAQuadTreeKernel() {
		const glm::ivec2 index{ getLaunchIndex() };

		if (isOutside(index, simulation.rendering.windowSize))
		{
			return;
		}

		Ray ray{ createRay(index) };
		BoxHit hit{ traceRayBoxes<State, false>(ray) };

		glm::vec3 col = simulation.rendering.materialColors[hit.material];

		if (hit.hit) {
			PhongBRDF brdf;
			brdf.position = hit.pos;
			brdf.normal = hit.normal;
			brdf.diffuseReflectance = col;
			brdf.alpha = 1.f;
			brdf.shininess = 40.f;
			brdf.F = 0.f;
			brdf.specularReflectance = glm::vec3(0);

			col = linearToSRGB(applyReinhardToneMap(shadePhongBRDFBoxes<State>(brdf) + simulation.rendering.uniforms->exposure * brdf.F * simulation.rendering.uniforms->ambientLight.luminance));
		}

		col = glm::clamp(col, 0.f, 1.f);
		uchar4 val = uchar4(col.x * 255u, col.y * 255u, col.z * 255u, 255u);
		surf2Dwrite(val, simulation.rendering.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

__global__ void raymarchDDAQuadTreeLevelKernel(int level) {
		const glm::ivec2 index{ getLaunchIndex() };

		if (isOutside(index, simulation.rendering.windowSize))
		{
			return;
		}

		Ray ray{ createRay(index) };
		ray.o += simulation.rendering.gridOffset;

		const glm::vec3 bmin = glm::vec3(0.f, -FLT_MAX, 0.f);
		const glm::vec3 bmax = glm::vec3(2.f * simulation.rendering.gridOffset.x, FLT_MAX, 2.f * simulation.rendering.gridOffset.z);

		bool intersect = intersection(bmin, bmax, ray.o, ray.i_dir, ray.t);
		ray.t.x = glm::max(ray.t.x, 0.f);

		BoxHit hit;

		int steps = 0;
		if (intersect) {
			// DDA prep
			DDAState state;
			calculateDDAState(state, ray, level);
			// DDA
			while (true) {
				steps++;

				hit.t = FLT_MAX;
				intersectSceneAsBoxes(state, ray, hit, level);

				if (hit.hit) {
					resolveBoxHit(hit, state, ray);
					break;
				}
				
				if (advanceDDA(state, ray, level)) break;

			}
		}

		hit.pos -= simulation.rendering.gridOffset;

		const glm::vec3 bgCol = glm::vec3(0);
		glm::vec3 col = /*glm::vec3(0.005f * steps);*/ 0.5f + 0.5f * hit.normal;
		col = hit.hit ? col : bgCol;



		col = glm::clamp(col, 0.f, 1.f);
		uchar4 val = uchar4(col.x * 255u, col.y * 255u, col.z * 255u, 255u);
		surf2Dwrite(val, simulation.rendering.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

__device__ __forceinline__ float getVolume(const glm::vec3& p, const float radius, const float radiusG) {
	const glm::vec3 boxBmin = p - radius;
	const glm::vec3 boxBmax = p + radius;

	const glm::vec2 pG = glm::vec2(p.x * simulation.rGridScale, p.z * simulation.rGridScale);

	float volume = 0.f;

	int y = glm::clamp(pG.y - radiusG, 0.f, float(simulation.gridSize.y));
	const float endY = glm::clamp(pG.y + radiusG, 0.f, float(simulation.gridSize.y));
	const float endX = glm::clamp(pG.x + radiusG, 0.f, float(simulation.gridSize.x));
	for (; y < endY; ++y) {
		int x = glm::clamp(pG.x - radiusG, 0.f, float(simulation.gridSize.x));
		for (; x < endX; ++x) {
			glm::ivec2 currentCell = glm::ivec2(x, y);

			//if (isOutside(currentCell, simulation.gridSize)) {
			//	continue;
			//}

			int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };
						
			float floor = -FLT_MAX;

			const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
			const glm::vec2 bmax2D = bmin2D + simulation.gridScale;

			for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
				const float totalTerrainHeight = terrainHeights[BEDROCK] + terrainHeights[SAND] + terrainHeights[WATER];

				if (totalTerrainHeight <= boxBmin.y) {
					floor = terrainHeights[CEILING];
					continue;
				}

				volume += intersectionVolume(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), boxBmin, boxBmax);

				floor = terrainHeights[CEILING];
				if (floor >= boxBmax.y) break;
			}
		}
	}
	return volume;
}

// How far can we step when computing the volume overlap of columns to an AABB of side length s? (see Arches paper definition of the implicit surface)
// My derivation:
// based on the last overlap, we calculate the volume change necessary to hit the surface (f(p) = 0)
// Moving along our ray, we retain part of the previous volume, remove some and add new volume.
// The fastest change would occur when moving diagonally through the AABB.
// Worst case, all the "old" volume was previously empty and all new volume is filled.
// The new volume gained is calculated as new_volume = s^3 - (s - distance/sqrt(3))^3.
// Solving for distance we get distance = 2*s*sqrt(3) - sqrt(3) (s + (s^3 - new_volume)^(1/3))
// However, for distances closer to s, the worst case changes to new_volume =  s^3 - (s - distance) * s * s 
// in which case distance = new_volume / (s * s)
// Using the necessary volume change from above as new_volume, we can calculate the safe distance.
// since f(p) = (2*i(p)/(s^3)) - 1, where i(p) is the volume overlap, given current volume X with f(p) > 0
// then the required volume Y is 0.5s^3, so the new volume gained is simply |X - (0.5s^3)|
__device__ __forceinline__ float calculateSafeStep(float volume, float targetVolume, float boxSize) {
	// TODO Precompute boxsize values?
	const float requiredVolumeGain = glm::abs(volume - targetVolume);
	const float boxSize2 = boxSize * boxSize;
	const float boxSize3 = boxSize2 * boxSize;
	const float d1 = requiredVolumeGain / boxSize2;
	constexpr float sqrt3 = std::numbers::sqrt3_v<float>;
	const float d2 = 2.f * boxSize * sqrt3 - sqrt3 * (boxSize + pow(boxSize3 - requiredVolumeGain, 1.f / 3.f));
	return glm::max(d1, d2);
}

template <class State>
__global__ void raymarchDDAQuadTreeSmoothKernel() {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.rendering.windowSize))
	{
		return;
	}

	Ray ray{ createRay(index) };
	ray.o += simulation.rendering.gridOffset;

	const glm::vec3 bmin = glm::vec3(0.f, -FLT_MAX, 0.f);
	const glm::vec3 bmax = glm::vec3(2.f * simulation.rendering.gridOffset.x, FLT_MAX, 2.f * simulation.rendering.gridOffset.z);

	bool intersect = intersection(bmin, bmax, ray.o, ray.i_dir, ray.t);
	ray.t.x = glm::max(ray.t.x, 0.f);

	BoxHit hit;

	int currentLevel = MAX_QUADTREE_LAYER;

	int steps = 0;
	if (intersect) {
		// DDA prep
		State state;
		calculateDDAState(state, ray, currentLevel);

		const float radius = simulation.gridScale * simulation.rendering.smoothingRadiusInCells;
		const float targetVolume = simulation.rendering.surfaceVolumePercentage * 8.f * radius * radius * radius;
		// DDA
		while (true) {
			steps++;

			// Simpler alternative to backtracking, no stack in state needed, has better performance
			if constexpr (std::is_same_v<State, DDAMissState>) {
				const int currentMissCount = currentLevel >= 0 ? simulation.rendering.missCount : simulation.rendering.fineMissCount;
				if ((currentLevel < MAX_QUADTREE_LAYER) && state.miss == currentMissCount) {
					currentLevel++;
					calculateDDAState(state, ray, currentLevel);
					continue;
				}
			}

			if (currentLevel < 0) {
				const glm::vec3 p = ray.o + (ray.t.x - radius) * ray.dir;
				const float volume = getVolume(p, radius, simulation.rendering.smoothingRadiusInCells);

				if (volume > 0.99f * targetVolume) {
					hit.hit = true;
					hit.t = ray.t.x - radius;
					hit.pos = p;
					// TODO: keep normal using smoother terrain?
					const float e = simulation.rendering.normalSmoothingFactor * radius;
					const float eG = simulation.rendering.normalSmoothingFactor * simulation.rendering.smoothingRadiusInCells;
					const float volumeN = simulation.rendering.normalSmoothingFactor != 1.f ? getVolume(p, e, eG) : volume;
					hit.normal = glm::normalize(glm::vec3(
						volumeN - getVolume(p + glm::vec3(e, 0.f, 0.f), e, eG),
						volumeN - getVolume(p + glm::vec3(0.f, e, 0.f), e, eG),
						volumeN - getVolume(p + glm::vec3(0.f, 0.f, e), e, eG)
					));
					break;
				}
				hit.t = volume; // Remember volume for step
			}
			else {
				intersectQuadTreeColumns(state, ray, hit, currentLevel);
			}

			if (hit.hit && (currentLevel >= 0)) {
				// Remember Exit for Backtracking
				if constexpr (std::is_same_v<State, DDAExitLevelState>) {
					const bool axis = state.T.x >= state.T.y;
					state.exitLevels[currentLevel] = state.T[axis];
				}
				// Hit coarse geometry in QuadTree. Move to finer level.
				ray.t.x = glm::max(ray.t.x, hit.t);
				currentLevel--;
				if (currentLevel >= 0) calculateDDAState(state, ray, currentLevel);
				hit.hit = false;
				continue;
			}

			// Backtracking - very delicate, easy to implement wrong
			if constexpr (std::is_same_v<State, DDAExitLevelState>) {
				bool movedUp = false;
				while ((currentLevel < MAX_QUADTREE_LAYER) && (ray.t.x >= state.exitLevels[currentLevel + 1] - bigEpsilon)) {
					currentLevel++;
					movedUp = true;
				}
				if (movedUp) {
					const float oldRay = ray.t.x;
					ray.t.x = glm::max(ray.t.x + bigEpsilon, state.exitLevels[currentLevel] + bigEpsilon);
					if (ray.t.y < ray.t.x) break;
					calculateDDAState(state, ray, currentLevel);
					ray.t.x = oldRay;
					continue;
				}
			}

			if (currentLevel < 0) {
				const float step = calculateSafeStep(hit.t, targetVolume, 2.f * radius);
				ray.t.x += step;
				if (ray.t.y < ray.t.x) break;
				if constexpr (std::is_same_v<State, DDAMissState>) {
					state.miss++;
				}
			}
			else {
				if (advanceDDA(state, ray, currentLevel)) break;
			}

		}
	}

	hit.pos -= simulation.rendering.gridOffset;

	const glm::vec3 bgCol = glm::vec3(0);
	glm::vec3 col = /*glm::vec3(0.005f * steps);*/ 0.5f + 0.5f * hit.normal;
	col = hit.hit ? col : bgCol;



	col = glm::clamp(col, 0.f, 1.f);
	uchar4 val = uchar4(col.x * 255u, col.y * 255u, col.z * 255u, 255u);
	surf2Dwrite(val, simulation.rendering.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}


__global__ void raymarchSmoothKernel() {
		const glm::ivec2 index{ getLaunchIndex() };

		if (isOutside(index, simulation.rendering.windowSize))
		{
			return;
		}

		const glm::vec3 ro = simulation.rendering.camPos;
		const glm::vec3 pW = simulation.rendering.lowerLeft + (index.x + 0.5f) * simulation.rendering.rightVec + (index.y + 0.5f) * simulation.rendering.upVec;
		const glm::vec3 r_dir = glm::normalize(pW - ro);
		const glm::vec3 inv_r_dir = 1.0f / r_dir;
		const auto inf_mask = glm::isinf(inv_r_dir);
		const glm::vec3 i_r_dir = glm::vec3(
			inf_mask.x ? 0.f : inv_r_dir.x,
			inf_mask.y ? 0.f : inv_r_dir.y,
			inf_mask.z ? 0.f : inv_r_dir.z
		);

		// Danger: potential infinite loop for rays that never leave xz bounds
		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.gridScale * float(simulation.gridSize.x), -FLT_MAX, -0.5f * simulation.gridScale * float(simulation.gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.gridScale * float(simulation.gridSize.x), FLT_MAX, 0.5f * simulation.gridScale * float(simulation.gridSize.y));
		glm::vec2 t;
		bool intersect = intersection(bmin, bmax, ro, i_r_dir, t);
		t.x = glm::max(t.x, 0.f);

		int steps = 0;
		bool hit = false;
		glm::vec3 p = glm::vec3(0);
		glm::vec3 n = glm::vec3(0);

		if (intersect) {
			// prep
			const glm::vec3 roGrid = glm::vec3(ro.x - bmin.x, ro.y, ro.z - bmin.z);

			const float radius = simulation.gridScale * simulation.rendering.smoothingRadiusInCells;
			const float targetVolume = simulation.rendering.surfaceVolumePercentage * 8.f * radius * radius * radius;

			// raymarching with box-filter for implicit surface
			while (t.x <= t.y) {
				steps++;
				p = roGrid + t.x * r_dir;
				const float volume = getVolume(p, radius, simulation.rendering.smoothingRadiusInCells);

				if (volume > 0.99f * targetVolume) {
					hit = true;
					// TODO: keep normal using smoother terrain?
					const float e = simulation.rendering.normalSmoothingFactor * radius;
					const float eG = simulation.rendering.normalSmoothingFactor * simulation.rendering.smoothingRadiusInCells;
					const float volumeN = simulation.rendering.normalSmoothingFactor != 1.f ? getVolume(p, e, eG) : volume;
					n = glm::normalize(glm::vec3(
						getVolume(p + glm::vec3(e, 0.f, 0.f), e, eG) - volumeN,
						getVolume(p + glm::vec3(0.f, e, 0.f), e, eG) - volumeN,
						getVolume(p + glm::vec3(0.f, 0.f, e), e, eG) - volumeN
					));
					break;
				}


				const float step = calculateSafeStep(volume, targetVolume, 2.f * radius);
				t.x += step;
			}
		}

		//normal = hit ? glm::vec3(1) : glm::vec3(-1);

		const glm::vec3 bgCol = glm::vec3(0);
		glm::vec3 col = 0.5f + 0.5f * n;
		col = hit ? col : bgCol;



		col = glm::clamp(col, 0.f, 1.f);
		uchar4 val = uchar4(col.x * 255u, col.y * 255u, col.z * 255u, 255u);
		surf2Dwrite(val, simulation.rendering.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

__global__ void buildQuadTreeFirstLayer() {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.quadTree[0].gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.quadTree[0].gridSize) };
	int maxOLayers = 0;
	const float radius = simulation.rendering.smoothingRadiusInCells * simulation.gridScale;

	struct Neighbor {
		glm::ivec2 index;
		int flatIndex;
		int layerCount;
		bool outside;
	} neighbors[4];

	for (int y = 0; y <= 1; ++y) {
		for (int x = 0; x <= 1; ++x) {
			const int oIndex = 2 * y + x;
			
			neighbors[oIndex].index = 2 * index + glm::ivec2(x, y);
			neighbors[oIndex].outside = isOutside(neighbors[oIndex].index, simulation.gridSize);
			if (neighbors[oIndex].outside)
			{
				continue;
			}

			neighbors[oIndex].flatIndex = flattenIndex(neighbors[oIndex].index, simulation.gridSize);
			neighbors[oIndex].layerCount = simulation.layerCounts[neighbors[oIndex].flatIndex];
			maxOLayers = glm::max(maxOLayers, neighbors[oIndex].layerCount);
		}
	}
	const char layerCount = glm::min(simulation.quadTree[0].maxLayerCount, maxOLayers);

	glm::vec4 heights;
	// Merge layer by layer, bottom to top
	for (int layer{ 0 }; layer < layerCount; ++layer) {
		heights = glm::vec4(-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX);
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}
				const auto oTerrainHeights = glm::cuda_cast(simulation.heights[neighbors[oIndex].flatIndex + simulation.layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				const float solidHeight = oTerrainHeights[BEDROCK] + oTerrainHeights[SAND];
				const float fullHeight = solidHeight + oTerrainHeights[WATER];
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], fullHeight + radius);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[CEILING] - radius);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], solidHeight + radius);
				heights[QAIR] = glm::min(heights[QAIR], fullHeight - radius);
			}
		}
		simulation.quadTree[0].heights[flatIndex + simulation.quadTree[0].layerStride * layer] = glm::cuda_cast(heights);
	}

	simulation.quadTree[0].layerCounts[flatIndex] = layerCount;

	// Merge remaining columns into topmost column
	for (int layer{ simulation.quadTree[0].maxLayerCount - 1 }; layer < maxOLayers; ++layer) {
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.heights[neighbors[oIndex].flatIndex + simulation.layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				const float solidHeight = oTerrainHeights[BEDROCK] + oTerrainHeights[SAND];
				const float fullHeight = solidHeight + oTerrainHeights[WATER];
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], fullHeight + radius);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[CEILING] - radius);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], solidHeight + radius);
				heights[QAIR] = glm::min(heights[QAIR], fullHeight - radius);
			}
		}
	}

	if (maxOLayers > layerCount) {
		simulation.quadTree[0].heights[flatIndex + simulation.quadTree[0].layerStride * (layerCount - 1)] = glm::cuda_cast(heights);
	}
	// TODO: Compact tree (merge overlapping columns)
}

__global__ void buildQuadTreeLayer(int i) {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.quadTree[i].gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.quadTree[i].gridSize) };
	int maxOLayers = 0;

		struct Neighbor {
		glm::ivec2 index;
		int flatIndex;
		int layerCount;
		bool outside;
	} neighbors[4];

	for (int y = 0; y <= 1; ++y) {
		for (int x = 0; x <= 1; ++x) {
			const int oIndex = 2 * y + x;
			
			neighbors[oIndex].index = 2 * index + glm::ivec2(x, y);
			neighbors[oIndex].outside = isOutside(neighbors[oIndex].index, simulation.quadTree[i - 1].gridSize);
			if (neighbors[oIndex].outside)
			{
				continue;
			}

			neighbors[oIndex].flatIndex = flattenIndex(neighbors[oIndex].index, simulation.quadTree[i - 1].gridSize);
			neighbors[oIndex].layerCount = simulation.quadTree[i - 1].layerCounts[neighbors[oIndex].flatIndex];
			maxOLayers = glm::max(maxOLayers, neighbors[oIndex].layerCount);
		}
	}
	const char layerCount = glm::min(simulation.quadTree[i].maxLayerCount, maxOLayers);

	glm::vec4 heights;
	// Merge layer by layer, bottom to top
	for (int layer{ 0 }; layer < layerCount; ++layer) {
		heights = glm::vec4(-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX);
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.quadTree[i - 1].heights[neighbors[oIndex].flatIndex + simulation.quadTree[i - 1].layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], oTerrainHeights[QFULLHEIGHT]);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[QCEILING]);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], oTerrainHeights[QSOLIDHEIGHT]);
				heights[QAIR] = glm::min(heights[QAIR], oTerrainHeights[QAIR]);
			}
		}
		simulation.quadTree[i].heights[flatIndex + simulation.quadTree[i].layerStride * layer] = glm::cuda_cast(heights);
	}
	simulation.quadTree[i].layerCounts[flatIndex] = layerCount;

	// Merge remaining columns into topmost column
	for (int layer{ simulation.quadTree[i].maxLayerCount - 1 }; layer < maxOLayers; ++layer) {
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.quadTree[i - 1].heights[neighbors[oIndex].flatIndex + simulation.quadTree[i - 1].layerStride * layer]);
				// No Padding needed here, since base level has it already
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], oTerrainHeights[QFULLHEIGHT]);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[QCEILING]);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], oTerrainHeights[QSOLIDHEIGHT]);
				heights[QAIR] = glm::min(heights[QAIR], oTerrainHeights[QAIR]);
			}
		}
	}

	if (maxOLayers > layerCount) {
		simulation.quadTree[i].heights[flatIndex + simulation.quadTree[i].layerStride * (layerCount - 1)] = glm::cuda_cast(heights);
	}
	// TODO: Compact tree (merge overlapping columns)
}

void buildQuadTree(const std::vector<Launch>& launch) {
	// first layer (special)
	CU_CHECK_KERNEL(buildQuadTreeFirstLayer << <launch[0].gridSize, launch[0].blockSize >> > ());
	// remaining layers
	for (int i = 1; i < launch.size(); ++i) {
		CU_CHECK_KERNEL(buildQuadTreeLayer << <launch[i].gridSize, launch[i].blockSize >> > (i));
	}
}

void raymarchTerrain(const Launch& launch, bool useInterpolation, int missCount, int debugLayer) {
	if (useInterpolation) {
		if (missCount < 0) {
			CU_CHECK_KERNEL(raymarchSmoothKernel << <launch.gridSize, launch.blockSize >> > ());
		}
		else if (missCount == 0) {
			CU_CHECK_KERNEL(raymarchDDAQuadTreeSmoothKernel<DDAExitLevelState> << <launch.gridSize, launch.blockSize >> > ());
		}
		else if (missCount >= 3) {
			CU_CHECK_KERNEL(raymarchDDAQuadTreeSmoothKernel<DDAMissState> << <launch.gridSize, launch.blockSize >> > ());
		}
		else {
			CU_CHECK_KERNEL(raymarchDDAQuadTreeSmoothKernel<DDAState> << <launch.gridSize, launch.blockSize >> > ());
		}
	}
	else {
		if (debugLayer < -1) {
			if (missCount == 0) {
				CU_CHECK_KERNEL(raymarchDDAQuadTreeKernel<DDAExitLevelState> << <launch.gridSize, launch.blockSize >> > ());
			}
			else if (missCount >= 3) {
				CU_CHECK_KERNEL(raymarchDDAQuadTreeKernel<DDAMissState> << <launch.gridSize, launch.blockSize >> > ());
			}
			else {
				CU_CHECK_KERNEL(raymarchDDAQuadTreeKernel<DDAState> << <launch.gridSize, launch.blockSize >> > ());
			}
		}
		else {
			CU_CHECK_KERNEL(raymarchDDAQuadTreeLevelKernel << <launch.gridSize, launch.blockSize >> > (debugLayer));
		}
	}
}
}
}