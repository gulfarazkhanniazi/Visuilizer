/**
 * The third geometry front-end: a real mesh.
 *
 * The flat-photo path inverts a homography and the panorama path casts a ray,
 * both to answer the same question -- where is this pixel, in metres, on the
 * surface? A glTF room already knows: the fragment has a world position. What
 * it does not have is which two of the three world axes to use, and that is
 * what the normal decides.
 *
 * Box mapping (pick the plane facing away from the dominant axis of the
 * normal) rather than the model's own UVs, deliberately: UVs are authored for
 * whatever texture the model shipped with and carry no real-world scale, so a
 * 600 mm tile laid against them would be 600 mm only by luck. World space is
 * metres by construction, which is the whole premise of the renderer.
 */
import { TILE_CORE } from './tileCore.glsl.js';

export const meshVertexShader = /* glsl */ `
out vec3 vWorld;
out vec3 vNrm;

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vNrm = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

export const meshFragmentShader = /* glsl */ `
precision highp float;
precision highp int;

in vec3 vWorld;
in vec3 vNrm;
out vec4 outColor;

uniform vec3  uLightDir;     // direction *towards* the key light
uniform float uAmbient;      // floor of the lighting term
uniform vec3  uOriginOffset; // shifts the pattern origin onto the room

${TILE_CORE}

void main() {
  vec3 n = normalize(vNrm);
  vec3 a = abs(n);
  vec3 p3 = vWorld - uOriginOffset;

  // Box mapping. A floor or ceiling reads its plane from XZ; a wall from
  // whichever of XY / ZY is square-on to it. Y stays vertical in both wall
  // cases, so a 300x600 tile stands up the way it would on site.
  vec2 world;
  if (a.y >= max(a.x, a.z))      world = vec2(p3.x, p3.z);
  else if (a.x >= a.z)           world = vec2(p3.z, p3.y);
  else                           world = vec2(p3.x, p3.y);

  vec2 dWdx = dFdx(world);
  vec2 dWdy = dFdy(world);

  // A modelled room has real lights rather than a photograph's baked ones, so
  // the lighting plate is computed here instead of sampled. Feeding it through
  // the same lum channel keeps one shading path for all three front-ends.
  float diff = max(dot(n, normalize(uLightDir)), 0.0);
  float sky = 0.5 + 0.5 * n.y;
  float lit = uAmbient + (1.0 - uAmbient) * (0.72 * diff + 0.28 * sky);
  vec2 lum = vec2(lit, lit);

  vec4 lay = shadeSurface(world, dWdx, dWdy, lum);
  if (lay.a <= 0.003) discard;
  outColor = vec4(lay.rgb, lay.a * uOpacity);
}
`;
