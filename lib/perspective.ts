import { Quad } from "./types";

const REGION_VERTEX_SRC = `#version 300 es
in vec2 aUnit;
uniform mat3 uHomography;
out vec2 vUnit;
void main() {
  vec3 clip = uHomography * vec3(aUnit, 1.0);
  gl_Position = vec4(clip.x, clip.y, 0.0, clip.z);
  vUnit = aUnit;
}
`;

const REGION_FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec2 vUnit;
uniform sampler2D uTile;
uniform sampler2D uMask;
uniform sampler2D uRoom;
uniform vec2 uRoomResolution;
uniform vec2 uRepeat;
uniform vec2 uOffset;
uniform float uMeanIntensity;
uniform float uShadingStrength;
uniform vec2 uShadingRange;
out vec4 outColor;
void main() {
  vec2 tileUv = vUnit * uRepeat + uOffset;
  vec4 tileColor = texture(uTile, tileUv);
  vec2 roomUv = vec2(gl_FragCoord.x, uRoomResolution.y - gl_FragCoord.y) / uRoomResolution;
  // The mask itself is a soft alpha matte, not a binary in/out decision (see _guided_filter_alpha
  // in the analysis service) — a boundary pixel is genuinely some fraction surface and some
  // fraction not, so it's blended proportionally via alpha rather than snapped to fully-painted or
  // fully-original. The discard here is purely a performance skip for texels that are essentially
  // zero coverage, not a coverage decision — that decision already happened upstream, at the
  // photo's true resolution, and shouldn't be re-made here at a coarser, aliased cutoff.
  float coverage = texture(uMask, roomUv).r;
  if (coverage < 0.02) discard;

  // Extract shading from original photo
  vec4 roomColor = texture(uRoom, roomUv);
  float roomIntensity = dot(roomColor.rgb, vec3(0.299, 0.587, 0.114));

  // local intensity relative to region's average intensity
  float shading = roomIntensity / max(uMeanIntensity, 0.05);

  // Bounded so a blown-out window highlight or a near-black shadow can't crush the design toward
  // white/black — that reads as "recolored by the original surface" rather than as a real material
  // lit naturally. The bound is per-surface (uShadingRange) rather than one figure for both,
  // because the two surfaces fail in opposite directions: a floor's realism lives almost entirely
  // in the contact shadows under furniture, while a wall mostly needs its corners to stay distinct
  // without picking up whatever colour cast the old paint had.
  shading = clamp(shading, uShadingRange.x, uShadingRange.y);

  // Apply shading strength blend
  shading = mix(1.0, shading, uShadingStrength);

  outColor = vec4(tileColor.rgb * shading, coverage);
}
`;

const PHOTO_VERTEX_SRC = `#version 300 es
in vec2 aPos;
in vec2 aUv;
out vec2 vUv;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
  vUv = aUv;
}
`;

const PHOTO_FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uRoom;
uniform vec2 uRoomTexSize;
uniform float uBicubic;
out vec4 outColor;

// Catmull-Rom. Chosen over the B-spline cubic that most "textureBicubic" snippets implement,
// because that one is *blurrier* than bilinear - it is a smoothing filter. Catmull-Rom
// interpolates through the samples and slightly overshoots at edges, which is what makes a
// magnified photo read as sharp rather than soft.
float crWeight(float x) {
  x = abs(x);
  if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
  if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
  return 0.0;
}

vec4 sampleCatmullRom(sampler2D tex, vec2 uv, vec2 texSize) {
  vec2 coord = uv * texSize - 0.5;
  vec2 base = floor(coord);
  vec2 f = coord - base;
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int j = -1; j <= 2; ++j) {
    for (int i = -1; i <= 2; ++i) {
      float w = crWeight(float(i) - f.x) * crWeight(float(j) - f.y);
      vec2 p = (base + vec2(float(i), float(j)) + 0.5) / texSize;
      acc += texture(tex, p) * w;
      wsum += w;
    }
  }
  return acc / max(wsum, 1e-5);
}

void main() {
  // Only when the photo is being magnified. Minifying wants the mipmap chain instead, and 16 taps
  // of a reconstruction filter would just be a slower way to alias.
  outColor = uBicubic > 0.5 ? sampleCatmullRom(uRoom, vUv, uRoomTexSize) : texture(uRoom, vUv);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${log}`);
  }
  return shader;
}

function linkProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${log}`);
  }
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  return program;
}

/** Maps the unit square (0,0)-(1,0)-(1,1)-(0,1) onto `dst` (already in clip/NDC space) via a projective transform. */
function unitSquareToQuadHomography(dst: Quad): Float32Array {
  const [{ x: X0, y: Y0 }, { x: X1, y: Y1 }, { x: X2, y: Y2 }, { x: X3, y: Y3 }] = dst;

  const dx1 = X1 - X2;
  const dx2 = X3 - X2;
  const dx3 = X0 - X1 + X2 - X3;
  const dy1 = Y1 - Y2;
  const dy2 = Y3 - Y2;
  const dy3 = Y0 - Y1 + Y2 - Y3;

  const denom = dx1 * dy2 - dx2 * dy1;
  let g = 0;
  let h = 0;
  if (Math.abs(dx3) > 1e-9 || Math.abs(dy3) > 1e-9) {
    g = (dx3 * dy2 - dx2 * dy3) / denom;
    h = (dx1 * dy3 - dx3 * dy1) / denom;
  }

  const a = X1 - X0 + g * X1;
  const b = X3 - X0 + h * X3;
  const c = X0;
  const d = Y1 - Y0 + g * Y1;
  const e = Y3 - Y0 + h * Y3;
  const f = Y0;

  // column-major for WebGL's uniformMatrix3fv
  return new Float32Array([a, d, g, b, e, h, c, f, 1]);
}

function imageToNdc(quad: Quad, imgWidth: number, imgHeight: number): Quad {
  return quad.map((p) => ({
    x: (p.x / imgWidth) * 2 - 1,
    y: 1 - (p.y / imgHeight) * 2,
  })) as Quad;
}

export interface RenderRegion {
  quad: Quad;
  texture: WebGLTexture;
  maskTexture: WebGLTexture;
  repeatX: number;
  repeatY: number;
  offsetX: number;
  offsetY: number;
  meanIntensity: number;
  /** How much of the photo's own light and shadow to carry into the design, 0-1. */
  shadingStrength: number;
  /** Lower/upper bound on that shading, as a multiplier of the surface's mean brightness. */
  shadingRange: [number, number];
}

export class RoomCompositor {
  private gl: WebGL2RenderingContext;
  private regionProgram: WebGLProgram;
  private photoProgram: WebGLProgram;
  private unitBuffer: WebGLBuffer;
  private photoBuffer: WebGLBuffer;
  private roomTexture: WebGLTexture | null = null;
  private imgWidth = 0;
  private imgHeight = 0;
  /** The photo texture's own pixel size, which is the source's, not the canvas'. */
  private roomTexWidth = 0;
  private roomTexHeight = 0;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not supported in this browser");
    this.gl = gl;

    this.regionProgram = linkProgram(gl, REGION_VERTEX_SRC, REGION_FRAGMENT_SRC);
    this.photoProgram = linkProgram(gl, PHOTO_VERTEX_SRC, PHOTO_FRAGMENT_SRC);

    const unit = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
    const unitBuffer = gl.createBuffer();
    if (!unitBuffer) throw new Error("Failed to create buffer");
    gl.bindBuffer(gl.ARRAY_BUFFER, unitBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, unit, gl.STATIC_DRAW);
    this.unitBuffer = unitBuffer;

    // fullscreen quad: pos.xy + uv.xy interleaved
    const photoQuad = new Float32Array([
      -1, -1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0,
    ]);
    const photoBuffer = gl.createBuffer();
    if (!photoBuffer) throw new Error("Failed to create buffer");
    gl.bindBuffer(gl.ARRAY_BUFFER, photoBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, photoQuad, gl.STATIC_DRAW);
    this.photoBuffer = photoBuffer;
  }

  get imageSize() {
    return { width: this.imgWidth, height: this.imgHeight };
  }

  /**
   * `width`/`height` stay the photo's own pixel space, because that is the space every quad and
   * mask is expressed in. `renderWidth`/`renderHeight` size the drawing buffer independently.
   *
   * They used to be the same thing, which quietly capped output quality at the source photo's
   * resolution: a 547px-wide stock photo produced a 547px-wide canvas that CSS then stretched to
   * ~890px on screen, so the browser upscaled the finished render and softened every tile edge
   * and grout line in it. The photo itself can't gain detail it never had, but the tile texture is
   * full-resolution art being minified into that small buffer, and the mask is delivered at up to
   * 2600px - so both have real detail to spend on a bigger buffer. Rendering at display size
   * instead of source size is what lets the design read as crisp material rather than a blur.
   */
  setRoomImage(
    image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
    width: number,
    height: number,
    renderWidth?: number,
    renderHeight?: number,
  ) {
    const gl = this.gl;
    this.imgWidth = width;
    this.imgHeight = height;
    this.canvas.width = Math.max(1, Math.round(renderWidth ?? width));
    this.canvas.height = Math.max(1, Math.round(renderHeight ?? height));

    // The source's own pixel size — a photo can be far larger or smaller than either the canvas
    // or the coordinate space the masks use, and the sampler needs the real texel grid.
    this.roomTexWidth =
      (image as HTMLImageElement).naturalWidth || (image as HTMLCanvasElement).width || width;
    this.roomTexHeight =
      (image as HTMLImageElement).naturalHeight || (image as HTMLCanvasElement).height || height;

    if (!this.roomTexture) this.roomTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.roomTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    // Mipmaps so a photo larger than the canvas is minified by averaging rather than by point
    // sampling every nth texel, which is what made a big phone photo shimmer along fine detail.
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  createTileTexture(image: HTMLImageElement | HTMLCanvasElement | ImageBitmap): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error("Failed to create texture");
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    return texture;
  }

  createMaskTexture(data: Uint8Array, width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error("Failed to create texture");
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  render(regions: RenderRegion[]) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (this.roomTexture) {
      gl.useProgram(this.photoProgram);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.photoBuffer);
      const posLoc = gl.getAttribLocation(this.photoProgram, "aPos");
      const uvLoc = gl.getAttribLocation(this.photoProgram, "aUv");
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.roomTexture);
      gl.uniform1i(gl.getUniformLocation(this.photoProgram, "uRoom"), 0);
      gl.uniform2f(
        gl.getUniformLocation(this.photoProgram, "uRoomTexSize"),
        this.roomTexWidth,
        this.roomTexHeight,
      );
      // Bicubic only pays off when magnifying; below 1:1 the mipmap chain is the right filter.
      const magnifying = this.roomTexWidth > 0 && this.canvas.width > this.roomTexWidth;
      gl.uniform1f(gl.getUniformLocation(this.photoProgram, "uBicubic"), magnifying ? 1 : 0);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    }

    if (regions.length === 0) return;

    gl.useProgram(this.regionProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.unitBuffer);
    const unitLoc = gl.getAttribLocation(this.regionProgram, "aUnit");
    gl.enableVertexAttribArray(unitLoc);
    gl.vertexAttribPointer(unitLoc, 2, gl.FLOAT, false, 0, 0);

    const homographyLoc = gl.getUniformLocation(this.regionProgram, "uHomography");
    const repeatLoc = gl.getUniformLocation(this.regionProgram, "uRepeat");
    const offsetLoc = gl.getUniformLocation(this.regionProgram, "uOffset");
    const roomResLoc = gl.getUniformLocation(this.regionProgram, "uRoomResolution");
    const meanIntensityLoc = gl.getUniformLocation(this.regionProgram, "uMeanIntensity");
    const shadingStrengthLoc = gl.getUniformLocation(this.regionProgram, "uShadingStrength");
    const shadingRangeLoc = gl.getUniformLocation(this.regionProgram, "uShadingRange");
    gl.uniform2f(roomResLoc, this.canvas.width, this.canvas.height);

    for (const region of regions) {
      const ndcQuad = imageToNdc(region.quad, this.imgWidth, this.imgHeight);
      const homography = unitSquareToQuadHomography(ndcQuad);
      gl.uniformMatrix3fv(homographyLoc, false, homography);
      gl.uniform2f(repeatLoc, region.repeatX, region.repeatY);
      gl.uniform2f(offsetLoc, region.offsetX, region.offsetY);
      gl.uniform1f(meanIntensityLoc, region.meanIntensity);
      // Set per surface by the caller — a floor and a wall need very different amounts of the
      // photo's original light carried through. See `SHADING` in RoomCanvas.
      gl.uniform1f(shadingStrengthLoc, region.shadingStrength);
      gl.uniform2f(shadingRangeLoc, region.shadingRange[0], region.shadingRange[1]);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, region.texture);
      gl.uniform1i(gl.getUniformLocation(this.regionProgram, "uTile"), 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.roomTexture);
      gl.uniform1i(gl.getUniformLocation(this.regionProgram, "uRoom"), 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, region.maskTexture);
      gl.uniform1i(gl.getUniformLocation(this.regionProgram, "uMask"), 2);

      gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    }
  }

  toDataURL(): string {
    return this.canvas.toDataURL("image/png");
  }
}
