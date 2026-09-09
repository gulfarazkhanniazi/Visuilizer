/**
 * Shaders for the flat-photo renderer.
 *
 * The geometry step is a homography: a flat surface in a photo is a plane, and
 * the map from that plane's metric coordinates to pixels is exactly a 3x3
 * projective transform. Inverting it per pixel gives a position in METRES,
 * which the shared tiling core in tileCore.glsl.js turns into laid tile.
 *
 * Written for WebGL2 / GLSL ES 3.00 so textureGrad is available.
 */
import { TILE_CORE } from './tileCore.glsl.js';

// RawShaderMaterial prepends nothing, so the built-in attributes three supplies
// on a PlaneGeometry have to be declared by hand.
export const fullscreenVertexShader = /* glsl */ `
precision highp float;

in vec3 position;
in vec2 uv;

out vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const surfaceVertexShader = fullscreenVertexShader;

export const surfaceFragmentShader = /* glsl */ `
precision highp float;
precision highp int;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uBlur;       // .r = blurred luminance, .g = raw luminance
uniform sampler2D uMask;       // .r = this surface's coverage, antialiased
uniform vec2  uResolution;     // photo size in pixels
uniform mat3  uHinv;           // undistorted pixels -> metres on the plane
uniform float uK1;             // lens distortion, division model; 0 = rectilinear

${TILE_CORE}

/**
 * Undo the lens's barrel distortion.
 *
 * A homography can only model a pinhole camera, so on a wide-angle phone photo
 * the tile rows bow away from straight near the frame edge. The division model
 * (r_u = r_d / (1 + k1 * r_d^2)) inverts in closed form, which is the whole
 * reason to prefer it here: the shader has an observed pixel and needs the
 * ideal one, which is exactly the direction Brown's polynomial cannot go.
 */
vec2 undistort(vec2 px) {
  if (uK1 == 0.0) return px;
  vec2 c = uResolution * 0.5;
  float s = length(uResolution) * 0.5;
  vec2 d = (px - c) / s;
  return c + (px - c) / (1.0 + uK1 * dot(d, d));
}

void main() {
  float cover = texture(uMask, vUv).r;
  if (cover <= 0.002) discard;

  vec2 px = undistort(vUv * uResolution);

  vec3 h = uHinv * vec3(px, 1.0);
  if (abs(h.z) < 1e-7) discard;
  vec2 world = h.xy / h.z;

  // Analytic screen-space derivatives of the plane coordinate. Deriving them
  // from the homography (rather than dFdx of a value that jumps at every
  // joint) is what keeps distant tiles from aliasing into noise.
  vec2 raw = vUv * uResolution;
  vec2 dpx = vec2(max(abs(dFdx(raw.x)), 1e-4), max(abs(dFdy(raw.y)), 1e-4));
  vec3 hx = uHinv * vec3(undistort(raw + vec2(dpx.x, 0.0)), 1.0);
  vec3 hy = uHinv * vec3(undistort(raw + vec2(0.0, dpx.y)), 1.0);
  vec2 dWdx = hx.xy / hx.z - world;
  vec2 dWdy = hy.xy / hy.z - world;

  vec2 lum = texture(uBlur, vUv).rg;
  vec4 lay = shadeSurface(world, dWdx, dWdy, lum);
  if (lay.a <= 0.001) discard;
  outColor = vec4(lay.rgb, lay.a * cover * uOpacity);
}
`;

/**
 * Panorama compositing pass.
 *
 * Runs over the equirectangular panorama itself rather than the screen: every
 * texel is a direction, so we cast that ray against each surface's plane, keep
 * the nearest hit, and shade it. Because the nearest hit wins, occlusion is
 * handled by the geometry -- a 360 room needs no hand-drawn masks at all.
 *
 * Up to MAX_PLANES surfaces are packed into uniform arrays; anything a real
 * room needs (floor, four walls, ceiling) fits comfortably.
 */
export const panoFragmentShader = /* glsl */ `
precision highp float;
precision highp int;

in vec2 vUv;
out vec4 outColor;

#define MAX_PLANES 8

uniform sampler2D uPano;       // the original panorama
uniform sampler2D uBlur;       // .r blurred luminance, .g raw
uniform vec2  uResolution;

uniform int   uCount;
uniform vec3  uOrigin[MAX_PLANES];   // a point on the plane, metres
uniform vec3  uNormal[MAX_PLANES];
uniform vec3  uAxisU[MAX_PLANES];    // unit vectors spanning the plane
uniform vec3  uAxisV[MAX_PLANES];
uniform vec4  uExtent[MAX_PLANES];   // uMin, uMax, vMin, vMax in metres
uniform int   uActive;               // which surface this pass is drawing

${TILE_CORE}

/** Equirectangular texel -> unit direction. v = 0 is the bottom of the sphere. */
vec3 dirFromUv(vec2 uv) {
  float lon = (uv.x - 0.5) * 6.2831853072;
  float lat = (uv.y - 0.5) * 3.1415926536;
  float cl = cos(lat);
  return vec3(cl * sin(lon), sin(lat), -cl * cos(lon));
}

/** Distance along d to plane i, or -1 when it is behind or parallel. */
float hitPlane(int i, vec3 d) {
  float denom = dot(uNormal[i], d);
  if (abs(denom) < 1e-6) return -1.0;
  float t = dot(uNormal[i], uOrigin[i]) / denom;
  if (t <= 1e-4) return -1.0;
  vec3 p = d * t;
  vec3 rel = p - uOrigin[i];
  float a = dot(rel, uAxisU[i]);
  float b = dot(rel, uAxisV[i]);
  vec4 e = uExtent[i];
  if (a < e.x || a > e.y || b < e.z || b > e.w) return -1.0;
  return t;
}

/** Which surface a direction lands on, or -1 for none. Nearest hit wins. */
int nearestSurface(vec3 d, out float dist) {
  float best = 1e9;
  int hit = -1;
  for (int i = 0; i < MAX_PLANES; i++) {
    if (i >= uCount) break;
    float t = hitPlane(i, d);
    if (t > 0.0 && t < best) { best = t; hit = i; }
  }
  dist = best;
  return hit;
}

void main() {
  vec2 texel = vec2(1.0) / uResolution;

  // Coverage from four sub-samples. The nearest-hit test is binary, so without
  // this every plane boundary -- every corner and skirting line in the room --
  // stair-steps once the panorama is magnified onto the sphere.
  float cover = 0.0;
  float best = 1e9;
  bool haveCentre = false;
  vec3 d = dirFromUv(vUv);

  for (int k = 0; k < 4; k++) {
    vec2 o = vec2(k == 0 || k == 3 ? -0.25 : 0.25, k < 2 ? -0.25 : 0.25);
    vec3 ds = dirFromUv(vUv + o * texel);
    float t;
    if (nearestSurface(ds, t) == uActive) {
      cover += 0.25;
      if (!haveCentre) { d = ds; best = t; haveCentre = true; }
    }
  }
  if (cover <= 0.0) discard;

  vec3 p = d * best;
  vec3 rel = p - uOrigin[uActive];
  vec2 world = vec2(dot(rel, uAxisU[uActive]), dot(rel, uAxisV[uActive]));

  // Derivatives by re-casting one texel across and one down: the projection is
  // not affine, so this is the honest way to size the filter footprint.
  vec2 dWdx = world;
  vec2 dWdy = world;
  float tx = hitPlane(uActive, dirFromUv(vUv + vec2(texel.x, 0.0)));
  float ty = hitPlane(uActive, dirFromUv(vUv + vec2(0.0, texel.y)));
  if (tx > 0.0) {
    vec3 q = dirFromUv(vUv + vec2(texel.x, 0.0)) * tx - uOrigin[uActive];
    dWdx = vec2(dot(q, uAxisU[uActive]), dot(q, uAxisV[uActive])) - world;
  } else dWdx = vec2(0.001);
  if (ty > 0.0) {
    vec3 q = dirFromUv(vUv + vec2(0.0, texel.y)) * ty - uOrigin[uActive];
    dWdy = vec2(dot(q, uAxisU[uActive]), dot(q, uAxisV[uActive])) - world;
  } else dWdy = vec2(0.001);

  vec2 lum = texture(uBlur, vUv).rg;
  vec4 lay = shadeSurface(world, dWdx, dWdy, lum);
  if (lay.a <= 0.001) discard;
  outColor = vec4(lay.rgb, lay.a * cover * uOpacity);
}
`;

/**
 * Separable blur that builds the lighting plate.
 * Pass 0 extracts luminance and blurs horizontally, keeping the raw value in
 * .g; pass 1 blurs .r vertically and carries .g through untouched.
 */
export const blurFragmentShader = /* glsl */ `
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSrc;
uniform vec2  uTexel;
uniform vec2  uDir;
uniform float uRadius;
uniform float uPass;

float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  bool first = uPass < 0.5;
  vec4 centre = texture(uSrc, vUv);
  float raw = first ? lum(centre.rgb) : centre.g;

  float total = 0.0;
  float wsum = 0.0;
  for (int i = -10; i <= 10; i++) {
    float fi = float(i);
    float w = exp(-0.5 * (fi / 4.0) * (fi / 4.0));
    vec2 uv = vUv + uDir * uTexel * (fi * uRadius / 10.0);
    vec4 s = texture(uSrc, uv);
    total += (first ? lum(s.rgb) : s.r) * w;
    wsum += w;
  }
  outColor = vec4(total / wsum, raw, 0.0, 1.0);
}
`;
