#include "simulation.hpp"

#include <numbers>

#include <cuda_fp16.h>

namespace geo {
	namespace device {

constexpr int MAX_QUADTREE_LAYER = geo::NUM_QUADTREE_LAYERS - 1;
constexpr int AIR = CEILING;

constexpr float3 WATER_ABSORPTION{ 0.2f, 0.15f, 0.1f };

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

struct SmoothHit {
	float t{ FLT_MAX };
	bool hit{ false };
	bool hit_air{ false };
	float min_volume_diff{ FLT_MAX };
	glm::vec3 materials{ 0.f };
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

struct PbrBRDF
{
	glm::vec3 diffuseReflectance;
	float roughness;
	float NdotV;
	float F0;
	float Fr;
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

__device__ __forceinline__ float DistributionGGX(const glm::vec3 &N, const glm::vec3& H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = glm::max(dot(N, H), 0.0f);
	float NdotH2 = NdotH * NdotH;

	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
	denom = glm::pi<float>() * denom * denom;

	return num / denom;
}

__device__ __forceinline__ float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;

	float num = NdotV;
	float denom = NdotV * (1.0f - k) + k;

	return num / denom;
}

__device__ __forceinline__ float GeometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness)
{
	float NdotV = glm::max(dot(N, V), 0.0f);
	float NdotL = glm::max(dot(N, L), 0.0f);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

__device__ __forceinline__ float getAttenuation(const float distance, const float range)
{
	float ratio = distance / range;
	return 1.f / (ratio * ratio + 1.0f);
}

__device__ __forceinline__ glm::vec3 getPointLightRadiance(const onec::RenderPipelineUniforms::PointLight pointLight, const glm::vec3 direction, const float distance)
{
	return getAttenuation(distance, pointLight.range) * pointLight.intensity;
}

__device__ __forceinline__ glm::vec3 getSpotLightRadiance(const onec::RenderPipelineUniforms::SpotLight spotLight, const glm::vec3 direction, const float distance)
{
	const float cutOff = glm::dot(spotLight.direction, direction);

	if (cutOff < spotLight.outerCutOff)
	{
		return glm::vec3(0.0f);
	}

	const float attenuation = glm::min((cutOff - spotLight.outerCutOff) / (spotLight.innerCutOff - spotLight.outerCutOff + epsilon), 1.0f) * getAttenuation(distance, spotLight.range);

	return attenuation * spotLight.intensity;
}

__device__ __forceinline__ glm::vec3 getDirectionalLightRadiance(const onec::RenderPipelineUniforms::DirectionalLight directionalLight, const glm::vec3 direction)
{
	return directionalLight.luminance;
}

__device__ __forceinline__ float fresnelSchlick(float cosTheta, float F0)
{
	return F0 + (1.0f - F0) * glm::pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

template<bool WaterMode = false>
__device__ __forceinline__ glm::vec2 fresnelSchlickRoughness(float cosTheta, float F0, float roughness)
{
	if (WaterMode) 
	{
		// For total internal reflections, pre-filled with constants for water-to-air intersection
		constexpr float inv_eta2 = 1.33f * 1.33f;
		float SinT2 = inv_eta2 * (1.0f - cosTheta * cosTheta);
		if (SinT2 > 1.0f)
		{
			return glm::vec2(F0 + (glm::max(1.0f - roughness, F0) - F0) * glm::pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f), 1.f);
		}
		cosTheta = sqrt(1.0f - SinT2);
	}

    return F0 + (glm::vec2(glm::max(1.0f - roughness, F0), 1.0f) - F0) * glm::pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}  

template<class Hit>
__device__ __forceinline__ glm::vec3 evaluatePbrBRDF(const PbrBRDF& pbrBRDF, const Hit& hit, const glm::vec3& L, const glm::vec3& V)
{
	const glm::vec3 H = normalize(V + L);
	float F  = fresnelSchlick(glm::max(dot(H, V), 0.0f), pbrBRDF.F0);
	float NDF = DistributionGGX(hit.normal, H, pbrBRDF.roughness);
	float G = GeometrySmith(hit.normal, V, L, pbrBRDF.roughness);
	float numerator = NDF * G * F;
	float denominator = 4.0f * glm::max(dot(hit.normal, V), 0.0f) * glm::max(dot(hit.normal, L), 0.0f) + bigEpsilon;
	glm::vec3 specular = glm::clamp(glm::vec3(numerator / denominator), glm::vec3(0.f), glm::vec3(1.f));

	float NdotL = glm::max(dot(hit.normal, L), 0.0f);
	return ((1.f - F) * pbrBRDF.diffuseReflectance * rPi + specular) * NdotL;
}

__device__ __forceinline__ bool intersection(const glm::vec3& b_min, const glm::vec3& b_max, const glm::vec3& r_o, const glm::vec3& inv_r_dir, glm::vec2& t) {
    glm::vec3 t1 = (b_min - r_o) * inv_r_dir;
    glm::vec3 t2 = (b_max - r_o) * inv_r_dir;

    glm::vec3 vtmin = glm::min(t1, t2);
    glm::vec3 vtmax = glm::max(t1, t2);

    t.x = glm::max(vtmin.x, glm::max(vtmin.y, vtmin.z));
    t.y = glm::min(vtmax.x, glm::min(vtmax.y, vtmax.z));

	return t.y >= t.x && t.y >= 0.f;
}

__device__ __forceinline__ float intersectionVolume(const glm::vec3& b_min0, const glm::vec3& b_max0, const glm::vec3& b_min1, const glm::vec3& b_max1) {
	const glm::vec3 b_min = glm::max(b_min0, b_min1);
	const glm::vec3 b_max = glm::min(b_max0, b_max1);
	const glm::vec3 d = glm::max(b_max - b_min, glm::vec3(0));

	return d.x * d.y * d.z;
}

__device__ __forceinline__ glm::vec3 intersectionVolumeBounds(const glm::vec3& b_min0, const glm::vec3& b_max0, const glm::vec3& b_min1, const glm::vec3& b_max1) {
	const glm::vec3 b_min = glm::max(b_min0, b_min1);
	const glm::vec3 b_max = glm::min(b_max0, b_max1);
	const glm::vec3 d = glm::max(b_max - b_min, glm::vec3(0));

	return glm::vec3(d.x * d.y * d.z, b_min.y, b_max.y);
}

__device__ __forceinline__ float intersectionVolumeFullBounds(glm::vec3& b_min, glm::vec3& b_max, const glm::vec3& b_min1, const glm::vec3& b_max1) {
	b_min = glm::max(b_min, b_min1);
	b_max = glm::min(b_max, b_max1);
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

template <class State, bool WaterMode = false>
// Ignores Water AND Air in WaterMode
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
		bool intersect = intersection(glm::vec3(state.bmin2D.x, floor, state.bmin2D.y), glm::vec3(state.bmax2D.x, terrainHeights[WaterMode ? SAND : WATER], state.bmax2D.y), ray.o, ray.i_dir, tempT);

		if (intersect && tempT.x < ray.t.y) {
			hit.hit = true;
			break;
		}

		floor = terrainHeights[CEILING];
	}
}

template <class State, bool WaterMode = false>
// Intersects with Air but not Water in WaterMode
__device__ __forceinline__ void intersectColumns(const State& state, const Ray& ray, BoxHit& hit) {
	int flatIndex{ flattenIndex(state.currentCell, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	float floor = -FLT_MAX;

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
		terrainHeights[SAND] += terrainHeights[BEDROCK];
		terrainHeights[WATER] += terrainHeights[SAND];
		const float totalTerrainHeight = terrainHeights[WaterMode ? SAND : WATER];

		glm::vec2 tempT;
		bool intersect = intersection(glm::vec3(state.bmin2D.x, floor, state.bmin2D.y), glm::vec3(state.bmax2D.x, totalTerrainHeight, state.bmax2D.y), ray.o, ray.i_dir, tempT);

		if (intersect && tempT.x < hit.t) {
			hit.hit = true;
			hit.boxHeights = glm::vec2(floor, totalTerrainHeight);
			hit.t = tempT.x;
			const float height = ray.o.y + hit.t * ray.dir.y;
			hit.material = WaterMode ? AIR : WATER;
			for (int i = SAND; i >= BEDROCK; --i) {
				if (height <= terrainHeights[i] + bigEpsilon) hit.material = i;
			}
		}

		floor = terrainHeights[WaterMode ? WATER : CEILING];
	}
	if constexpr (WaterMode) {
		glm::vec2 tempT;
		bool intersect = intersection(glm::vec3(state.bmin2D.x, floor, state.bmin2D.y), glm::vec3(state.bmax2D.x, FLT_MAX, state.bmax2D.y), ray.o, ray.i_dir, tempT);
		if (intersect && tempT.x < hit.t) {
			hit.hit = true;
			hit.boxHeights = glm::vec2(floor, FLT_MAX);
			hit.t = tempT.x;
			const float height = ray.o.y + hit.t * ray.dir.y;
			hit.material = AIR;
		}
	}
}

template <class State, class Hit, bool Shadow = false, bool WaterMode = false>
__device__ __forceinline__ void intersectQuadTreeColumns(const State& state, const Ray& ray, Hit& hit, int currentLevel) {
	int flatIndex{ flattenIndex(state.currentCell, simulation.quadTree[currentLevel].gridSize) };
	const int layerCount{ simulation.quadTree[currentLevel].layerCounts[flatIndex] };

	float floor = -FLT_MAX;

	const glm::vec2 bmin2D = glm::vec2(state.currentCell) * simulation.quadTree[currentLevel].gridScale;
	const glm::vec2 bmax2D = bmin2D + simulation.quadTree[currentLevel].gridScale;

	int layer{ 0 };
	for (; layer < layerCount; ++layer, flatIndex += simulation.quadTree[currentLevel].layerStride)
	{
		const auto terrainHeights = glm::cuda_cast(simulation.quadTree[currentLevel].heights[flatIndex]);
		/*if constexpr (Shadow && SoftShadows) {
			const float radius = simulation.rendering.smoothingRadiusInCells * simulation.gridScale;
			terrainHeights[QFULLHEIGHT] += radius;
			terrainHeights[QCEILING] -= radius;
			terrainHeights[QSOLIDHEIGHT] += radius;
			terrainHeights[QAIR] -= radius;
		}*/
		const float totalTerrainHeight = terrainHeights[WaterMode ? QSOLIDHEIGHT : QFULLHEIGHT];

		glm::vec2 tempT;
		bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), ray.o, ray.i_dir, tempT);
		if (intersect && (tempT.y > ray.t.x)) {
			hit.hit = true;
			if constexpr(std::is_same_v<Hit, BoxHit>) hit.boxHeights = glm::vec2(floor, totalTerrainHeight);
			hit.t = glm::max(tempT.x, ray.t.x);
			break;
		}

		floor = terrainHeights[WaterMode ? (Shadow ? QCEILING : QAIR) : QCEILING];
	}
	if constexpr (WaterMode && !Shadow) {
		if (layer == layerCount) {
			glm::vec2 tempT;
			bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, FLT_MAX, bmax2D.y), ray.o, ray.i_dir, tempT);
			if (intersect && (tempT.y > ray.t.x)) {
				hit.hit = true;
				if constexpr(std::is_same_v<Hit, BoxHit>) hit.boxHeights = glm::vec2(floor, FLT_MAX);
				hit.t = glm::max(tempT.x, ray.t.x);
			}
		}
	}
}

template <class State, bool WaterMode = false>
__device__ __forceinline__ void intersectSceneAsBoxes(const State& state, const Ray& ray, BoxHit& hit, int currentLevel) {
	if (currentLevel < 0) {
		// Terrain column intersection
		intersectColumns<State, WaterMode>(state, ray, hit);
	}
	else {
		// QuadTree Column Intersection
		intersectQuadTreeColumns<State, BoxHit, false, WaterMode>(state, ray, hit, currentLevel);
	}
}

template <class State, bool WaterMode = false>
__device__ __forceinline__ void intersectSceneAsBoxesAny(const State& state, const Ray& ray, BoxHit& hit, int currentLevel) {
	if (currentLevel < 0) {
		// Terrain column intersection
		intersectColumnsAny<State, WaterMode>(state, ray, hit);
	}
	else {
		// QuadTree Column Intersection
		intersectQuadTreeColumns<State, BoxHit, true, WaterMode>(state, ray, hit, currentLevel);
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

__device__ __forceinline__ float calculateSafeStep(float volume, float targetVolume, float boxSize, float rBoxSize2, float boxSize3) {
	// TODO Precompute boxsize values?
	const float requiredVolumeGain = glm::abs(volume - targetVolume);
	//const float d1 = min_step + factor * requiredVolumeGain * simulation.rendering.rBoxSize2;// / boxSize2;
	const float d1 = requiredVolumeGain * rBoxSize2;
	//return d1;
	constexpr float sqrt3 = std::numbers::sqrt3_v<float>;
	const float d2 = 2.f * boxSize * sqrt3 - sqrt3 * (boxSize + pow(boxSize3 - requiredVolumeGain, 1.f / 3.f));
	return glm::min(d1, d2);
}

template<bool WaterMode = false>
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

			int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };
						
			float floor = -FLT_MAX;

			const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
			const glm::vec2 bmax2D = bmin2D + simulation.gridScale;

			int layer{ 0 };
			for (; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
				const float totalTerrainHeight = terrainHeights[BEDROCK] + terrainHeights[SAND] + (WaterMode ? 0.f : terrainHeights[WATER]);

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

//template<bool Shadow = false, bool WaterMode = false>
__device__ __forceinline__ glm::vec2 getTerrainAirVolume(const glm::vec3& p, const float radius, const float radiusG) {
	const glm::vec3 boxBmin = p - radius;
	const glm::vec3 boxBmax = p + radius;

	const glm::vec2 pG = glm::vec2(p.x * simulation.rGridScale, p.z * simulation.rGridScale);

	glm::vec2 volume = glm::vec2(0.f);
	//volume.y = 8.f * radius * radius * radius;
	//volume.y -= intersectionVolume(glm::vec3(0.f, boxBmin.y, 0.f), glm::vec3(simulation.gridScale * simulation.gridSize.x, boxBmax.y, simulation.gridScale * simulation.gridSize.y), boxBmin, boxBmax);

	int y = glm::clamp(pG.y - radiusG, 0.f, float(simulation.gridSize.y));
	const float endY = glm::clamp(pG.y + radiusG, 0.f, float(simulation.gridSize.y));
	const float endX = glm::clamp(pG.x + radiusG, 0.f, float(simulation.gridSize.x));
	for (; y < endY; ++y) {
		int x = glm::clamp(pG.x - radiusG, 0.f, float(simulation.gridSize.x));
		for (; x < endX; ++x) {
			glm::ivec2 currentCell = glm::ivec2(x, y);

			int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };

			glm::vec2 floor = glm::vec2(-FLT_MAX);

			const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
			const glm::vec2 bmax2D = bmin2D + simulation.gridScale;

			int layer{ 0 };
			for (; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
				const float totalTerrainHeight = terrainHeights[BEDROCK] + terrainHeights[SAND];
				floor.y = totalTerrainHeight + terrainHeights[WATER];

				if (terrainHeights[CEILING] <= boxBmin.y) {
					floor.x = terrainHeights[CEILING];
					continue;
				}

				volume.x += intersectionVolume(glm::vec3(bmin2D.x, floor.x, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), boxBmin, boxBmax);
				volume.y += intersectionVolume(glm::vec3(bmin2D.x, floor.y, bmin2D.y), glm::vec3(bmax2D.x, terrainHeights[CEILING], bmax2D.y), boxBmin, boxBmax);

				floor.x = terrainHeights[CEILING];
				if (floor.x >= boxBmax.y) break;
			}
		}
	}
	return volume;
}

__device__ __forceinline__ glm::vec3 getMaterialVolume(const glm::vec3& p, const float radius, const float radiusG, const glm::vec3& weight = glm::vec3(1.f)) {
	const glm::vec3 boxBmin = p - radius;
	const glm::vec3 boxBmax = p + radius;

	const glm::vec2 pG = glm::vec2(p.x * simulation.rGridScale, p.z * simulation.rGridScale);

	const glm::vec3 vWeight = weight;

	glm::vec4 volume = glm::vec4(0.f);

	int y = glm::clamp(pG.y - radiusG, 0.f, float(simulation.gridSize.y));
	const float endY = glm::clamp(pG.y + radiusG, 0.f, float(simulation.gridSize.y));
	const float endX = glm::clamp(pG.x + radiusG, 0.f, float(simulation.gridSize.x));
	for (; y < endY; ++y) {
		int x = glm::clamp(pG.x - radiusG, 0.f, float(simulation.gridSize.x));
		for (; x < endX; ++x) {
			glm::ivec2 currentCell = glm::ivec2(x, y);

			int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };
						
			float floor = -FLT_MAX;

			const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
			const glm::vec2 bmax2D = bmin2D + simulation.gridScale;

			int layer{ 0 };
			for (; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
				glm::vec3 materials = glm::vec3(0.f);
				materials.x = terrainHeights[BEDROCK];
				materials.y = terrainHeights[SAND] + materials.x;
				materials.z = terrainHeights[WATER] + materials.y;
				const float totalTerrainHeight = materials.z;

				if (totalTerrainHeight <= boxBmin.y) {
					floor = terrainHeights[CEILING];
					continue;
				}

				glm::vec3 intersection = intersectionVolumeBounds(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), boxBmin, boxBmax);
				const float interval = intersection.z - intersection.y;
				materials = glm::clamp((materials - intersection.y) / interval, 0.f, 1.f);
				materials.z -= materials.y;
				materials.y -= materials.x;
				volume += glm::vec4(vWeight * materials * intersection.x, intersection.x);

				floor = terrainHeights[CEILING];
				if (floor >= boxBmax.y) break;
			}
		}
	}
	const float i_sum = 1.f / (volume.x + volume.y + volume.z);
	const float i_sum2 = 1.f / (volume.x + volume.y + bigEpsilon);

	return glm::vec3(i_sum2 * volume.x, i_sum2 * volume.y, i_sum * volume.z);
}

template <class State, bool Shadow, bool WaterMode = false>
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

	if (intersect) {
		// DDA prep
		State state;
		calculateDDAState(state, ray, currentLevel);
		// DDA
		while (true) {

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
				intersectSceneAsBoxesAny<State, WaterMode>(state, ray, hit, currentLevel);
			}
			else {
				intersectSceneAsBoxes<State, WaterMode>(state, ray, hit, currentLevel);
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
				//ray.t.x = glm::max(ray.t.x, hit.t);
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

template <class State, bool Shadow = false, bool WaterMode = false, bool ForceNormal = false, bool SoftShadows = false>
__device__ __forceinline__ SmoothHit traceRaySmooth(Ray& ray, float bias = 0.f) {
	ray.o += simulation.rendering.gridOffset;

	const glm::vec3 bmin = glm::vec3(0.f, -2000.f, 0.f);
	const glm::vec3 bmax = glm::vec3(2.f * simulation.rendering.gridOffset.x, 2000.f, 2.f * simulation.rendering.gridOffset.z);

	glm::vec2 t;
	bool intersect = intersection(bmin, bmax, ray.o, ray.i_dir, t);
	ray.t.x = glm::max(ray.t.x, t.x);
	ray.t.y = glm::min(ray.t.y, t.y);
	if (ray.t.y <= ray.t.x) intersect = false;

	SmoothHit hit{};

	int currentLevel = MAX_QUADTREE_LAYER;
	const float radius = simulation.gridScale * simulation.rendering.smoothingRadiusInCells;
	const float rBoxSize2 = simulation.rendering.rBoxSize2;
	const float boxSize3 = simulation.rendering.boxSize3;
	const float targetVolume = (simulation.rendering.surfaceVolumePercentage + bias) * simulation.rendering.boxSize3;

	if (intersect) {
		// DDA prep
		State state;
		calculateDDAState(state, ray, currentLevel);


		// DDA
		while (true) {
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
				glm::vec3 p = ray.o + ray.t.x * ray.dir;// glm::max(ray.t.x - radius, 0.f)* ray.dir;
				float volume;
				bool air_hit = false;
				if constexpr (WaterMode && !Shadow) {
					glm::vec2 taVolume = getTerrainAirVolume(p, radius, simulation.rendering.smoothingRadiusInCells);
					if (taVolume.y > taVolume.x) {
						volume = taVolume.y;
						air_hit = true;
					}
					else {
						volume = taVolume.x;
					}
				}
				else {
					volume = getVolume<WaterMode>(p, radius, simulation.rendering.smoothingRadiusInCells);
				}

				if constexpr(Shadow && SoftShadows) hit.min_volume_diff = glm::min(hit.min_volume_diff, glm::max((targetVolume - volume) / glm::min(0.01f * simulation.rGridScale * ray.t.x, 0.4f), 0.f));
				if (volume >= 0.99f * targetVolume) {
					hit.hit = true;
					if constexpr (!Shadow || ForceNormal) {
						if (volume > targetVolume) {
							p -= calculateSafeStep(volume, targetVolume, 2.f * radius, rBoxSize2, boxSize3) * ray.dir;
						}
						else if (volume < targetVolume) {
							p += calculateSafeStep(volume, targetVolume, 2.f * radius, rBoxSize2, boxSize3) * ray.dir;
						}
						hit.pos = p;
						hit.materials = getMaterialVolume(hit.pos, radius, simulation.rendering.smoothingRadiusInCells);
						hit.hit_air = air_hit;
						hit.t = ray.t.x;// glm::max(ray.t.x - radius, 0.f);
						// TODO: keep normal using smoother terrain?
						const float e = simulation.rendering.normalSmoothingFactor * radius;
						const float eG = simulation.rendering.normalSmoothingFactor * simulation.rendering.smoothingRadiusInCells;
						if (hit.hit_air) {
							const float volumeN = getVolume(p, e, eG);
							hit.normal = glm::normalize(glm::vec3(
								getVolume(p + glm::vec3(e, 0.f, 0.f), e, eG) - volumeN,
								getVolume(p + glm::vec3(0.f, e, 0.f), e, eG) - volumeN,
								getVolume(p + glm::vec3(0.f, 0.f, e), e, eG) - volumeN
							));
						}
						else {
							const float volumeN = getVolume<WaterMode>(p, e, eG);
							hit.normal = glm::normalize(glm::vec3(
								volumeN - getVolume<WaterMode>(p + glm::vec3(e, 0.f, 0.f), e, eG),
								volumeN - getVolume<WaterMode>(p + glm::vec3(0.f, e, 0.f), e, eG),
								volumeN - getVolume<WaterMode>(p + glm::vec3(0.f, 0.f, e), e, eG)
							));
						}
					}
					break;
				}
				hit.t = volume; // Remember volume for step
			}
			else {
				intersectQuadTreeColumns<State, SmoothHit, Shadow, WaterMode>(state, ray, hit, currentLevel);
			}

			if (hit.hit && (currentLevel >= 0)) {
				// Remember Exit for Backtracking
				if constexpr (std::is_same_v<State, DDAExitLevelState>) {
					const bool axis = state.T.x >= state.T.y;
					state.exitLevels[currentLevel] = state.T[axis];
				}
				// Hit coarse geometry in QuadTree. Move to finer level.
				//ray.t.x = glm::max(ray.t.x, hit.t);
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
				const float step = calculateSafeStep(hit.t, targetVolume, 2.f * radius, rBoxSize2, boxSize3);
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
	if constexpr (WaterMode && !Shadow) {
		if (!hit.hit) {
			const glm::vec3 p = ray.o + ray.t.y * ray.dir;
			const float e = simulation.rendering.normalSmoothingFactor * radius;
			const float eG = simulation.rendering.normalSmoothingFactor * simulation.rendering.smoothingRadiusInCells;
			const float volumeN = getVolume(p, e, eG);

			if (volumeN > 0.f) {
				hit.hit = true;
				hit.hit_air = true;
				hit.pos = p;
				hit.materials = getMaterialVolume(hit.pos, radius, simulation.rendering.smoothingRadiusInCells);
				//if constexpr(WaterMode) hit.materials = glm::vec4(1.f, 0.f, 0.f, 1.f);
				hit.t = ray.t.y;
				hit.normal = glm::normalize(glm::vec3(
					getVolume(p + glm::vec3(e, 0.f, 0.f), e, eG) - volumeN,
					getVolume(p + glm::vec3(0.f, e, 0.f), e, eG) - volumeN,
					getVolume(p + glm::vec3(0.f, 0.f, e), e, eG) - volumeN
				));
			}
		}
	}
	ray.o -= simulation.rendering.gridOffset;
	hit.pos -= simulation.rendering.gridOffset;

	return hit;
}

template<class State, class Hit, bool WaterMode = false, bool SoftShadows = false>
__device__ __forceinline__ float getShadow(const Hit& hit, const glm::vec3& direction, float lightDistance, float bias = 0.f) {

	Ray ray{ createRay(hit.pos, direction) };
	if constexpr (std::is_same_v<Hit, BoxHit>) {
		ray.o += bigEpsilon * direction;
	}
	ray.t.y = lightDistance;
	Hit rHit;
	float shadow = 0.f;
	if constexpr (std::is_same_v<Hit, BoxHit>) {
		rHit = traceRayBoxes<State, true, WaterMode>(ray);
		shadow = !rHit.hit;
	}
	else {
		rHit = traceRaySmooth<State, true, WaterMode, false, SoftShadows>(ray, bias);
		if (SoftShadows) {
			shadow = /*rHit.hit ? 0.f :*/ glm::clamp(rHit.min_volume_diff * simulation.rendering.rBoxSize3, 0.f, 1.f);
		}
		else {
			shadow = !rHit.hit;
		}
	}
	return shadow;
}

__device__ __forceinline__ glm::vec3 getAmbientLightLuminance(const onec::RenderPipelineUniforms::AmbientLight& ambientLight, const PbrBRDF& pbrBRDF)
{
	return (rPi * pbrBRDF.diffuseReflectance) * ambientLight.luminance;
}

template<class State, class Hit, bool WaterMode = false, bool SoftShadows = false>
__device__ __forceinline__ glm::vec3 getPointLightLuminance(const onec::RenderPipelineUniforms::PointLight& pointLight, const PbrBRDF& pbrBRDF, const Hit& hit, const glm::vec3& direction, bool shadow, float bias = 0.f)
{
	const glm::vec3 lightVector = pointLight.position - hit.pos;
	const float lightDistance = length(lightVector);
	const glm::vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const glm::vec3 luminance = evaluatePbrBRDF(pbrBRDF, hit, lightDirection, direction) *
		getPointLightRadiance(pointLight, -lightDirection, lightDistance);

	if (shadow && luminance != glm::vec3(0.f)) {
		return luminance * getShadow<State, Hit, WaterMode, SoftShadows>(hit, lightDirection, lightDistance, bias);
	}

	return luminance;
}

template<class State, class Hit, bool WaterMode = false, bool SoftShadows = false>
__device__ __forceinline__ glm::vec3 getSpotLightLuminance(const onec::RenderPipelineUniforms::SpotLight& spotLight, const PbrBRDF& pbrBRDF, const Hit& hit, const glm::vec3& direction, bool shadow, float bias = 0.f)
{
	const glm::vec3 lightVector = spotLight.position - hit.pos;
	const float lightDistance = length(lightVector);
	const glm::vec3 lightDirection = lightVector / (lightDistance + epsilon);

	const glm::vec3 luminance = evaluatePbrBRDF(pbrBRDF, hit, lightDirection, direction) *
		getSpotLightRadiance(spotLight, -lightDirection, lightDistance);

	if (shadow && luminance != glm::vec3(0.f)) {
		return luminance * getShadow<State, Hit, WaterMode, SoftShadows>(hit, lightDirection, lightDistance, bias);
	}

	return luminance;
}

template<class State, class Hit, bool WaterMode = false, bool SoftShadows = false>
__device__ __forceinline__ glm::vec3 getDirectionalLightLuminance(const onec::RenderPipelineUniforms::DirectionalLight& directionalLight, const PbrBRDF& pbrBRDF, const Hit& hit, const glm::vec3& direction, bool shadow, float bias = 0.f)
{
	const glm::vec3 L = -directionalLight.direction;

	const glm::vec3 luminance = evaluatePbrBRDF(pbrBRDF, hit, L, direction) *
		getDirectionalLightRadiance(directionalLight, directionalLight.direction);

	if (shadow && luminance != glm::vec3(0.f)) {
		return luminance * getShadow<State, Hit, WaterMode, SoftShadows>(hit, L, FLT_MAX, bias);
	}

	return luminance;
}

template <class State, class Hit, bool WaterMode = false, bool SoftShadows = false>
__device__ __forceinline__ glm::vec3 shadePbrBRDF(const PbrBRDF& pbrBRDF, const Ray& ray, const Hit& hit, bool shadow, bool ao, float bias = 0.f, float ambient_weight = 1.f)
{
	const glm::vec3 viewDirection = -ray.dir;

	float sAo = 1.f;
	if (ao && ambient_weight > 0.f) {
		const float radius = simulation.rendering.aoRadius;
		const float radiusG = radius * simulation.rGridScale;
		const float volumePercent = 1.f - simulation.rendering.rAoBoxSize3 * getVolume<true>(simulation.rendering.gridOffset + hit.pos + radius * hit.normal, radius, radiusG);
		
		sAo = volumePercent * volumePercent;
	}

	glm::vec3 luminance(0.f);

	if (ambient_weight > 0.f) {
		glm::vec2 envBRDF = glm::cuda_cast(tex2D<float2>(simulation.rendering.integratedBRDFTexture, pbrBRDF.NdotV, pbrBRDF.roughness));

		luminance = sAo * ambient_weight * (
			(1.f - pbrBRDF.Fr) * getAmbientLightLuminance(simulation.rendering.uniforms->ambientLight, pbrBRDF)
			+ simulation.rendering.uniforms->ambientLight.luminance * (pbrBRDF.Fr * envBRDF.x + envBRDF.y)
			);
	}

	for (int i = 0; i < simulation.rendering.uniforms->pointLightCount; ++i)
	{
		luminance += getPointLightLuminance<State, Hit, WaterMode, SoftShadows>(simulation.rendering.uniforms->pointLights[i], pbrBRDF, hit, viewDirection, shadow, bias);
	}

	for (int i = 0; i < simulation.rendering.uniforms->spotLightCount; ++i)
	{
		luminance += getSpotLightLuminance<State, Hit, WaterMode, SoftShadows>(simulation.rendering.uniforms->spotLights[i], pbrBRDF, hit, viewDirection, shadow, bias);
	}

	for (int i = 0; i < simulation.rendering.uniforms->directionalLightCount; ++i)
	{
		luminance += getDirectionalLightLuminance<State, Hit, WaterMode, SoftShadows>(simulation.rendering.uniforms->directionalLights[i], pbrBRDF, hit, viewDirection, shadow, bias);
	}

	return luminance;
}


// nt for basic dielectric: 1.5 => R0 of 0.04 in Air; in Water: 0.009
// nt for water: 1.33 => R0 of 0.02 in Air; in Water when we hit Air: 0.02
template <class Hit, bool WaterMode = false>
__device__ __forceinline__ PbrBRDF getBRDF(const Ray& ray, Hit& hit, bool weight_diffuse = false, const glm::vec3& fallbackNormal = glm::vec3(0.f,1.f,0.f)) {
	PbrBRDF brdf;

	if (glm::any(glm::isnan(hit.normal))) {
		hit.normal = fallbackNormal;
	}

	brdf.NdotV = glm::max(glm::dot(-ray.dir, hit.normal), 0.0f);

	if constexpr (std::is_same_v<Hit, BoxHit>) {
		brdf.F0 = (hit.material == WATER || hit.material == AIR) ? 0.02f : (WaterMode ? 0.009f : 0.04f);
		brdf.roughness = (hit.material == WATER) ? 0.05f : 1.f;
		auto F = (WaterMode && hit.material == AIR) ? fresnelSchlickRoughness<WaterMode>(brdf.NdotV, brdf.F0, brdf.roughness) : fresnelSchlickRoughness(brdf.NdotV, brdf.F0, brdf.roughness);
		brdf.Fr = F.x;
		brdf.F = F.y;
		brdf.diffuseReflectance = (hit.material == WATER) ? glm::vec3(0.f) : simulation.rendering.materialColors[hit.material];
	}
	else {
		// if materials.z is exactly 1, some shading gets disabled for performance, this ensures it works properly
		hit.materials.z *= 1.f / (1.f - bigEpsilon);
		hit.materials.z = glm::min(hit.materials.z, 1.f);

		brdf.F0 = glm::mix((WaterMode ? 0.009f : 0.04f), 0.02f, hit.hit_air ? 1.f : hit.materials.z);
		brdf.roughness = 1.f - (WaterMode && !hit.hit_air ? 0.f : 0.95f * hit.materials.z);
		auto F = hit.hit_air ? fresnelSchlickRoughness<WaterMode>(brdf.NdotV, brdf.F0, brdf.roughness) : fresnelSchlickRoughness(brdf.NdotV, brdf.F0, brdf.roughness);
		brdf.Fr = F.x;
		brdf.F = F.y;
		brdf.diffuseReflectance = (weight_diffuse ? 1.f - hit.materials.z : 1.f) * (
			  hit.materials.x * simulation.rendering.materialColors[BEDROCK]
			+ hit.materials.y * simulation.rendering.materialColors[SAND]
			//+ hit.materials.z * simulation.rendering.materialColors[WATER]
			);
	}
	return brdf;
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
			PbrBRDF brdf{ getBRDF(ray, hit) };

			glm::vec3 reflection = glm::vec3(0.f);
			if (hit.material == WATER) {
				const glm::vec3 rDir = glm::normalize(glm::reflect(ray.dir, hit.normal));
				Ray rRay{ createRay(hit.pos + bigEpsilon * rDir, rDir) };
				BoxHit rHit{ traceRayBoxes<State, false>(rRay) };
				reflection = simulation.rendering.materialColors[rHit.material];

				if (rHit.hit) {
					PbrBRDF rBrdf{ getBRDF(rRay, rHit) };
					// Secondary reflections simply sample background
					reflection = shadePbrBRDF<State>(rBrdf, rRay, rHit, true, true);
				}
			}
			glm::vec3 refraction = glm::vec3(0.f);
			if (hit.material == WATER) {
				const glm::vec3 rDir = glm::normalize(glm::refract(ray.dir, hit.normal, 0.75f));
				Ray rRay{ createRay(hit.pos + bigEpsilon * rDir, rDir) };
				BoxHit rHit{ traceRayBoxes<State, false, true>(rRay) };
				refraction = simulation.rendering.materialColors[rHit.material];

				if (rHit.hit && rHit.material != AIR) {
					PbrBRDF rBrdf{ getBRDF(rRay, rHit) };
					// Secondary reflections simply sample background
					refraction = shadePbrBRDF<State, BoxHit, true>(rBrdf, rRay, rHit, true, true);
				}
			}
			col = shadePbrBRDF<State>(brdf, ray, hit, true, true) + glm::mix(refraction, reflection, brdf.F);
		}

		col = glm::clamp(linearToSRGB(applyReinhardToneMap(simulation.rendering.uniforms->exposure * col)), 0.f, 1.f);
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

template <class State>
__global__ void raymarchDDAQuadTreeSmoothKernel() {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.rendering.windowSize))
	{
		return;
	}

	Ray ray{ createRay(index) };

	bool waterRay = false;

	auto originMats = getMaterialVolume(ray.o + simulation.rendering.gridOffset, 0.1f * simulation.gridScale, 0.1f);
	if (originMats.z > 1.f - bigEpsilon) waterRay = true;

	SmoothHit hit = waterRay ? traceRaySmooth<State, false, true>(ray) : traceRaySmooth<State>(ray);


	const glm::vec3 bgCol = glm::vec3(0);

	glm::vec3 col = simulation.rendering.materialColors[AIR];

	if (hit.hit) {
		if (waterRay) {
			PbrBRDF brdf{ getBRDF<SmoothHit,true> (ray, hit) };

			glm::vec3 reflection = glm::vec3(0.f);
			if (hit.hit_air) {
				const glm::vec3 rDir = glm::normalize(glm::reflect(ray.dir, hit.normal));
				Ray rRay{ createRay(hit.pos, rDir) };
				SmoothHit rHit{ traceRaySmooth<State, false, true>(rRay, 0.01f) };
				reflection = simulation.rendering.materialColors[AIR];

				if (rHit.hit) {
					PbrBRDF rBrdf{ getBRDF<SmoothHit, true>(rRay, rHit) };
					// Secondary reflections simply sample background
					reflection = shadePbrBRDF<State, SmoothHit, true>(rBrdf, rRay, rHit, true, true, 0.02f, 1.f - rHit.materials.z);
					if (rHit.hit_air) reflection += rHit.materials.z * simulation.rendering.materialColors[AIR];
				}
			}
			glm::vec3 refraction = glm::vec3(0.f);

			col = shadePbrBRDF<State, SmoothHit, true, true>(brdf, ray, hit, true, true, 0.01f, 1.f - hit.materials.z);

			if (hit.hit_air) {
				const glm::vec3 rDir = glm::normalize(glm::refract(ray.dir, hit.normal, 1.33f));
				Ray rRay{ createRay(hit.pos, rDir) };

				SmoothHit rHit{ traceRaySmooth<State, true, true, true>(rRay, 0.01f) };
				refraction = simulation.rendering.materialColors[AIR];

				if (rHit.hit) {
					PbrBRDF rBrdf{ getBRDF(rRay, rHit) };
					// Secondary reflections simply sample background
					refraction = shadePbrBRDF<State>(rBrdf, rRay, rHit, true, true, 0.02f, 1.f - rHit.materials.z);
					refraction += rHit.materials.z * rBrdf.F * simulation.rendering.materialColors[AIR];
				}
			}
			col = col + hit.materials.z * ((1.f - brdf.F) * refraction + brdf.F * reflection);
		}
		else if (!hit.hit_air) {
			PbrBRDF brdf{ getBRDF(ray, hit, true) };

			glm::vec3 reflection = glm::vec3(0.f);
			if (hit.materials.z > 0.f) {
				const glm::vec3 rDir = glm::normalize(glm::reflect(ray.dir, hit.normal));
				Ray rRay{ createRay(hit.pos, rDir) };
				SmoothHit rHit{ traceRaySmooth<State, true, true, true>(rRay, 0.01f) };
				reflection = simulation.rendering.materialColors[AIR];

				if (rHit.hit) {
					PbrBRDF rBrdf{ getBRDF(rRay, rHit) };
					// Secondary reflections simply sample background
					reflection = shadePbrBRDF<State>(rBrdf, rRay, rHit, true, true, 0.02f, 1.f - rHit.materials.z);
					reflection += rHit.materials.z * rBrdf.F * simulation.rendering.materialColors[AIR];
				}
			}
			glm::vec3 refraction = glm::vec3(0.f);

			col = shadePbrBRDF<State, SmoothHit, false, true>(brdf, ray, hit, true, true, 0.01f, 1.f - hit.materials.z);

			if (hit.materials.z > 0.0f) {
				const glm::vec3 rDir = glm::normalize(glm::refract(ray.dir, hit.normal, 0.75f));
				Ray rRay{ createRay(hit.pos, rDir) };

				SmoothHit rHit{ traceRaySmooth<State, false, true>(rRay, 0.01f) };
				refraction = simulation.rendering.materialColors[AIR];

				if (rHit.hit) {
					PbrBRDF rBrdf{ getBRDF<SmoothHit, true>(rRay, rHit, false, hit.normal) };

					// Secondary reflections simply sample background
					refraction = shadePbrBRDF<State, SmoothHit, true>(rBrdf, rRay, rHit, true, true, 0.02f, 1.f - rHit.materials.z);
					if (rHit.hit_air) refraction += rHit.materials.z * (1.f - rBrdf.F) * simulation.rendering.materialColors[AIR];

					Ray uRay{ createRay(rHit.pos + 0.1f * glm::vec3(0.f,1.f, 0.f), glm::normalize(glm::vec3(-glm::sign(rHit.pos.x) * 0.01f, 1.f, -glm::sign(rHit.pos.z) * 0.01f)))};
					SmoothHit uHit{ traceRaySmooth<State, false, true>(uRay, 0.0f) };

					const float d = -glm::min((uHit.hit ? (uHit.hit_air ? uHit.t : rHit.t) : 0.f) + rHit.t, 2.f * rHit.t);
					refraction *= glm::exp(d * glm::cuda_cast(WATER_ABSORPTION));
				}
			}
			col = col + hit.materials.z * ((1.f - brdf.F) * refraction + brdf.F * reflection);
		}
	}

	col = glm::clamp(linearToSRGB(applyReinhardToneMap(simulation.rendering.uniforms->exposure * col)), 0.f, 1.f);
	//col = glm::clamp(glm::vec3(hit.steps * 0.01f), 0.f, 1.f);
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
			const float rBoxSize2 = simulation.rendering.rBoxSize2;
			const float boxSize3 = simulation.rendering.boxSize3;
			const float targetVolume = simulation.rendering.surfaceVolumePercentage * simulation.rendering.boxSize3;

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
						volumeN - getVolume(p + glm::vec3(e, 0.f, 0.f), e, eG),
						volumeN - getVolume(p + glm::vec3(0.f, e, 0.f), e, eG),
						volumeN - getVolume(p + glm::vec3(0.f, 0.f, e), e, eG)
					));
					break;
				}


				const float step = calculateSafeStep(volume, targetVolume, 2.f * radius, rBoxSize2, boxSize3);
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


struct Neighbor {
	glm::ivec2 index;
	int flatIndex;
	int layerCount;
	bool outside;
};

__device__ __forceinline__ Neighbor getNeighbor(const glm::ivec2& index, const glm::ivec2& offset) {
	Neighbor neigh;
	neigh.index = 2 * index + offset;
	neigh.outside = isOutside(neigh.index, simulation.gridSize);
	if (neigh.outside)
	{
		neigh.layerCount = 0;
		return neigh;
	}

	neigh.flatIndex = flattenIndex(neigh.index, simulation.gridSize);
	neigh.layerCount = simulation.layerCounts[neigh.flatIndex];
	return neigh;
}

__device__ __forceinline__ void mergeQuadTreeLayer(int treeLevel, int layerCount, int flatIndex) {
	if (layerCount > 1) {
		bool merge = false;
		int merge_layer = layerCount - 1;
		int write_offset = 0;
		glm::vec4 currHeights;
		for (int layer{ 0 }; layer < layerCount; ++layer) {
			currHeights = glm::cuda_cast(simulation.quadTree[treeLevel].heights[flatIndex + layer * simulation.quadTree[treeLevel].layerStride]);
			// 3 Intervals we care about: [CEILING - 1, FULLHEIGHT], [CEILING - 1, SOLIDHEIGHT], [AIR - 1, SOLIDHEIGHT]
			// SOLIDHEIGHT <= FULLHEIGHT
			// AIR <= CEILING
			// AIR <= FULLHEIGHT

			if (currHeights[QCEILING] <= currHeights[QSOLIDHEIGHT]) {
				// All 3 Intervals have overlap in this case, so we merge => set all values to values in above column
				if (!merge) {
					// First mergepoint found, remember layer
					merge_layer = layer;
					merge = true;
				} // else, no need to do anything, we are merging multiple layers consecutively
				write_offset++;
			}
			else {
				if (merge) {
					printf("[Tree %i] Commiting merge to layer %i from layer %i\n", treeLevel, merge_layer, layer);
					// Commit the merge
					simulation.quadTree[treeLevel].heights[flatIndex + merge_layer * simulation.quadTree[treeLevel].layerStride] = glm::cuda_cast(currHeights);
					merge = false;
				}
				else if (write_offset > 0) {
					simulation.quadTree[treeLevel].heights[flatIndex + (layer - write_offset) * simulation.quadTree[treeLevel].layerStride] = glm::cuda_cast(currHeights);
				}
			}
		}
		simulation.quadTree[treeLevel].layerCounts[flatIndex] = glm::max(layerCount - write_offset, 1);
		if (merge) {
			printf("[Tree %i] Uncommited merge\n", treeLevel);
			// Uncommited merge - should never happen if the basic datastructure rules aren't compromised somehow
			simulation.quadTree[treeLevel].heights[flatIndex + merge_layer * simulation.quadTree[treeLevel].layerStride] = glm::cuda_cast(currHeights);
		}
	}
	else {
		simulation.quadTree[treeLevel].layerCounts[flatIndex] = layerCount;
	}
}

__global__ void buildQuadTreeFirstLayer(bool interpolation) {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.quadTree[0].gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.quadTree[0].gridSize) };
	int maxOLayers = 0;

	Neighbor neighbors[4]; // cache the 2x2 neighbors, the larger neighborhood is dynamic so not cached

	int indexRadius = interpolation ? glm::ceil(simulation.rendering.smoothingRadiusInCells) : 0.f;
	float radius = interpolation ? simulation.gridScale * simulation.rendering.smoothingRadiusInCells : 0.f;

	for (int y = 0 - indexRadius; y <= 1 + indexRadius; ++y) {
		for (int x = 0 - indexRadius; x <= 1 + indexRadius; ++x) {
			const int oIndex = 2 * y + x;
			const bool validIndex = interpolation ? (y >= 0 && x >= 0 && y <= 1 && x <= 1) : true;
			const auto neigh = getNeighbor(index, glm::ivec2(x, y));
			if (validIndex) neighbors[oIndex] = neigh;
			
			maxOLayers = glm::max(maxOLayers, neigh.layerCount);
		}
	}
	const char layerCount = glm::min(simulation.quadTree[0].maxLayerCount, maxOLayers);


	glm::vec4 heights;
	// Merge layer by layer, bottom to top
	for (int layer{ 0 }; layer < layerCount; ++layer) {
		heights = glm::vec4(-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX);
		for (int y = 0 - indexRadius; y <= 1 + indexRadius; ++y) {
			for (int x = 0 - indexRadius; x <= 1 + indexRadius; ++x) {
				const int oIndex = 2 * y + x;
				const bool validIndex = interpolation ? (y >= 0 && x >= 0 && y <= 1 && x <= 1) : true;
				Neighbor neighbor = validIndex ? neighbors[oIndex] : getNeighbor(index, glm::ivec2(x, y));

				if (neighbor.outside || (neighbor.layerCount < (layer + 1))) {
					continue;
				}
				const auto oTerrainHeights = glm::cuda_cast(simulation.heights[neighbor.flatIndex + simulation.layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				const float solidHeight = oTerrainHeights[BEDROCK] + oTerrainHeights[SAND];
				const float fullHeight = solidHeight + oTerrainHeights[WATER];
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], fullHeight + radius) ;
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[CEILING] - radius);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], solidHeight + radius);
				heights[QAIR] = glm::min(heights[QAIR], fullHeight - radius);
			}
		}
		simulation.quadTree[0].heights[flatIndex + simulation.quadTree[0].layerStride * layer] = glm::cuda_cast(heights);
	}

	// Merge remaining columns into topmost column
	for (int layer{ simulation.quadTree[0].maxLayerCount - 1 }; layer < maxOLayers; ++layer) {
		for (int y = 0 - indexRadius; y <= 1 + indexRadius; ++y) {
			for (int x = 0 - indexRadius; x <= 1 + indexRadius; ++x) {
				const int oIndex = 2 * y + x;
				const bool validIndex = (y >= 0 && x >= 0 && y <= 1 && x <= 1);
				Neighbor neighbor = validIndex ? neighbors[oIndex] : getNeighbor(index, glm::ivec2(x, y));

				if (neighbor.outside || (neighbor.layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.heights[neighbor.flatIndex + simulation.layerStride * layer]);
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
	// Compact tree (merge overlapping columns)
	mergeQuadTreeLayer(0, layerCount, flatIndex);
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
	// Compact tree (merge overlapping columns)
	mergeQuadTreeLayer(i, layerCount, flatIndex);
}

__device__ __forceinline__ float GeometrySchlickGGXIBL(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0f;

    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
__device__ __forceinline__ float GeometrySmithIBL(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness)
{
    float NdotV = glm::max(dot(N, V), 0.0f);
    float NdotL = glm::max(dot(N, L), 0.0f);
    float ggx2 = GeometrySchlickGGXIBL(NdotV, roughness);
    float ggx1 = GeometrySchlickGGXIBL(NdotL, roughness);

    return ggx1 * ggx2;
}  

__device__ __forceinline__ glm::vec3 ImportanceSampleGGX(glm::vec2 Xi, glm::vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0f * glm::pi<float>() * Xi.x;
    float cosTheta = sqrt((1.0f - Xi.y) / (1.0f + (a*a - 1.0f) * Xi.y));
    float sinTheta = sqrt(1.0f - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    glm::vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    glm::vec3 up        = abs(N.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 tangent   = normalize(cross(up, N));
    glm::vec3 bitangent = cross(N, tangent);
	
    glm::vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return glm::normalize(sampleVec);
}  

__device__ __forceinline__ float RadicalInverse_VdC(unsigned int bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
}
// ----------------------------------------------------------------------------
__device__ __forceinline__ glm::vec2 Hammersley(unsigned int i, unsigned int N)
{
    return glm::vec2(float(i)/float(N), RadicalInverse_VdC(i));
}  

__device__ __forceinline__ glm::vec2 IntegrateBRDF(float NdotV, float roughness)
{
    glm::vec3 V;
    V.x = sqrt(1.0f - NdotV*NdotV);
    V.y = 0.0f;
    V.z = NdotV;

    float A = 0.0f;
    float B = 0.0f;

    glm::vec3 N = glm::vec3(0.0f, 0.0f, 1.0f);

    const unsigned int SAMPLE_COUNT = 1024u;
    for(unsigned int i = 0u; i < SAMPLE_COUNT; ++i)
    {
        glm::vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        glm::vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
        glm::vec3 L  = glm::normalize(2.0f * dot(V, H) * H - V);

        float NdotL = glm::max(L.z, 0.0f);
        float NdotH = glm::max(H.z, 0.0f);
        float VdotH = glm::max(dot(V, H), 0.0f);

        if(NdotL > 0.0f)
        {
            float G = GeometrySmithIBL(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0f - VdotH, 5.0f);

            A += (1.0f - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return glm::vec2(A, B);
}

__global__ void integrateBRDFKernel(glm::ivec2 size, glm::vec2 rSize) {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, size))
	{
		return;
	}

	glm::vec2 uv = rSize * (glm::vec2(index) + 0.5f);
	glm::vec2 val_f = IntegrateBRDF(uv.x, uv.y);

	half2 val{ val_f.x, val_f.y };
	ushort2 rawVal = *reinterpret_cast<ushort2*>(&val);
	surf2Dwrite(rawVal, simulation.rendering.integratedBRDFSurface, index.x * sizeof(ushort2), index.y, cudaBoundaryModeTrap);
}

void integrateBRDF(const Launch& launch, glm::ivec2 size) {
	CU_CHECK_KERNEL(integrateBRDFKernel << <launch.gridSize, launch.blockSize >> > (size, 1.f / glm::vec2(size)));
}

void buildQuadTree(const std::vector<Launch>& launch, bool useInterpolation) {
	// first layer (special)
	CU_CHECK_KERNEL(buildQuadTreeFirstLayer << <launch[0].gridSize, launch[0].blockSize >> > (useInterpolation));
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