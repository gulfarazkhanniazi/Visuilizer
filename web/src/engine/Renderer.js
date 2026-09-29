import * as THREE from 'three';
import {
  fullscreenVertexShader,
  surfaceFragmentShader,
  blurFragmentShader,
} from './surfaceShader.js';
import { surfaceHomography, invert3 } from './homography.js';
import { rasterizeMask, referenceLuminance } from './masks.js';
import {
  loadImage, buildTileAtlas, canvasTexture, imageTexture, imagePixels, hexToRgb,
} from './textures.js';
import { LAYOUT_BY_KEY, materialModelId, materialModel } from './layouts.js';

const FRAMES = ['left', 'right'];

/**
 * The room renderer.
 *
 * Pipeline per frame:
 *   photo  -> [blur plate]  (once per room)
 *   photo  -> frame target  (background)
 *   + one masked, homography-mapped, relit quad per visible surface
 *   frame targets -> screen, through a zoom/pan + compare-split pass
 */
export default class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    // All compositing happens on sRGB-encoded values, the way a photo editor's
    // multiply blend does. Letting three re-encode behind our back would double
    // up the transfer function and wash the relighting out. Set here rather
    // than at module scope so the 3D renderer, which does want colour
    // management, can make the opposite choice for its own scene.
    THREE.ColorManagement.enabled = false;
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.maxAniso = this.renderer.capabilities.getMaxAnisotropy();

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.PlaneGeometry(2, 2);

    this.room = null;
    this.photoTex = null;
    this.size = { w: 1, h: 1 };

    this.surfaces = new Map();       // name -> { mask, quad meta, hInv, refLevel }
    this.products = new Map();       // productId -> { texture, faces }
    this.state = { left: {}, right: {} };
    this.compare = { enabled: false, split: 0.5 };
    this.view = { zoom: 1, x: 0.5, y: 0.5 };
    this.highlight = { name: null, strength: 0 };

    this.frameTargets = {};
    this.lumTargets = [];
    this.blurRadius = 6;
    // Lens distortion coefficient for this room's photo. 0 is a rectilinear
    // camera, which is what every homography here assumes until told otherwise.
    this.k1 = 0;

    this._buildPasses();
    this._dirty = true;
    this._raf = null;
  }

  // ---------------------------------------------------------------- passes --

  _buildPasses() {
    this.blurMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: blurFragmentShader,
      uniforms: {
        uSrc: { value: null },
        uTexel: { value: new THREE.Vector2() },
        uDir: { value: new THREE.Vector2(1, 0) },
        uRadius: { value: 6 },
        uPass: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.blurMesh = new THREE.Mesh(this.quad, this.blurMat);

    this.presentMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: /* glsl */ `
        precision highp float;
        in vec2 vUv;
        out vec4 outColor;
        uniform sampler2D uLeft;
        uniform sampler2D uRight;
        uniform sampler2D uMaskHi;
        uniform float uCompare;
        uniform float uSplit;
        uniform float uHighlight;
        uniform vec2  uScale;      // fit the photo into the canvas
        uniform vec2  uPan;
        uniform float uZoom;
        uniform vec2  uCanvas;

        void main() {
          // Screen uv -> photo uv, honouring aspect-fit plus zoom and pan.
          vec2 uv = (vUv - 0.5) / uScale / uZoom + uPan;
          if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            outColor = vec4(0.055, 0.06, 0.07, 1.0);
            return;
          }
          vec3 col = texture(uLeft, uv).rgb;
          if (uCompare > 0.5) {
            float s = step(uSplit, vUv.x);
            col = mix(col, texture(uRight, uv).rgb, s);
            float d = abs(vUv.x - uSplit) * uCanvas.x;
            col = mix(vec3(1.0), col, smoothstep(0.0, 1.5, d - 0.5));
          }
          if (uHighlight > 0.001) {
            float m = texture(uMaskHi, uv).r;
            // A moving diagonal sheen reads as "this is the live surface"
            // without hiding the material underneath it.
            float sheen = 0.5 + 0.5 * sin((uv.x + uv.y) * 34.0);
            col = mix(col, mix(col, vec3(0.25, 0.62, 1.0), 0.42 + 0.18 * sheen), m * uHighlight);
          }
          outColor = vec4(col, 1.0);
        }
      `,
      uniforms: {
        uLeft: { value: null },
        uRight: { value: null },
        uMaskHi: { value: null },
        uCompare: { value: 0 },
        uSplit: { value: 0.5 },
        uHighlight: { value: 0 },
        uScale: { value: new THREE.Vector2(1, 1) },
        uPan: { value: new THREE.Vector2(0.5, 0.5) },
        uZoom: { value: 1 },
        uCanvas: { value: new THREE.Vector2(1, 1) },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.presentMesh = new THREE.Mesh(this.quad, this.presentMat);

    this.bgMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: /* glsl */ `
        precision highp float;
        in vec2 vUv;
        out vec4 outColor;
        uniform sampler2D uPhoto;
        void main() { outColor = vec4(texture(uPhoto, vUv).rgb, 1.0); }
      `,
      uniforms: { uPhoto: { value: null } },
      depthTest: false,
      depthWrite: false,
    });
    this.bgMesh = new THREE.Mesh(this.quad, this.bgMat);
  }

  _newSurfaceMaterial() {
    return new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: surfaceFragmentShader,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: {
        uBlur: { value: null },
        uMask: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uHinv: { value: new THREE.Matrix3() },
        uRotation: { value: 0 },
        uOffset: { value: new THREE.Vector2() },
        uTile: { value: null },
        uFaces: { value: 1 },
        uTileSize: { value: new THREE.Vector2(0.6, 0.6) },
        uTint: { value: new THREE.Vector3(1, 1, 1) },
        uMaterial: { value: 0 },
        uColor: { value: new THREE.Vector3(0.92, 0.89, 0.84) },
        uPlaneSize: { value: new THREE.Vector2(3, 3) },
        uK1: { value: 0 },
        uLayout: { value: 0 },
        uRandomFace: { value: 1 },
        uRandomRotate: { value: 0 },
        uGrout: { value: 0.002 },
        uGroutColor: { value: new THREE.Vector3(0.8, 0.8, 0.78) },
        uBevel: { value: 0.35 },
        uRefLevel: { value: 0.5 },
        uShade: { value: 1 },
        uDetail: { value: 0.6 },
        uGloss: { value: 0.25 },
        uOpacity: { value: 1 },
      },
    });
  }

  // ------------------------------------------------------------------ room --

  /**
   * Load a room: its photo, every surface's mask and homography, and the
   * lighting plate derived from the photo itself.
   */
  async setRoom(room, photoUrl) {
    const img = await loadImage(photoUrl);
    this.room = room;
    this.size = { w: img.naturalWidth, h: img.naturalHeight };
    this.k1 = Number(room.settings?.lensK1) || 0;

    this.photoTex?.dispose();
    this.photoTex = imageTexture(img);
    this.bgMat.uniforms.uPhoto.value = this.photoTex;

    this._allocTargets();
    this._buildLightingPlate();

    const px = imagePixels(img);
    for (const s of this.surfaces.values()) this._disposeSurface(s);
    this.surfaces.clear();

    for (const obj of room.objectList ?? []) {
      this.surfaces.set(obj.name, this._makeSurface(obj, px));
    }

    this._dirty = true;
    return this;
  }

  _makeSurface(obj, px) {
    const maskCanvas = rasterizeMask(obj.mask, this.size.w, this.size.h);
    const maskTex = canvasTexture(maskCanvas);

    // The mask was rasterised at photo resolution but luminance analysis runs
    // on a downscaled copy, so the polygon test needs the same scale factor.
    const scaled = scaleMask(obj.mask, px.scale);
    const refLevel = referenceLuminance(px.data, scaled, px.width, px.height);

    const H = surfaceHomography(obj.quad, obj.realSize.w, obj.realSize.h, this._lens());
    const hInv = H ? invert3(H) : null;

    const materials = {};
    for (const f of FRAMES) {
      const m = this._newSurfaceMaterial();
      m.uniforms.uBlur.value = this.lumTargets[1].texture;
      m.uniforms.uMask.value = maskTex;
      m.uniforms.uResolution.value.set(this.size.w, this.size.h);
      if (hInv) m.uniforms.uHinv.value.fromArray(hInv);
      m.uniforms.uRefLevel.value = refLevel;
      m.uniforms.uPlaneSize.value.set(obj.realSize.w, obj.realSize.h);
      m.uniforms.uK1.value = this.k1;
      materials[f] = m;
    }

    return {
      obj,
      maskCanvas,
      maskTex,
      refLevel,
      hInv,
      materials,
      meshes: Object.fromEntries(
        FRAMES.map((f) => [f, new THREE.Mesh(this.quad, materials[f])]),
      ),
    };
  }

  _disposeSurface(s) {
    s.maskTex?.dispose();
    for (const f of FRAMES) s.materials[f]?.dispose();
  }

  /** Recompute one surface's mask/homography after the studio edits it. */
  refreshSurface(name) {
    const s = this.surfaces.get(name);
    if (!s) return;
    const obj = s.obj;

    s.maskCanvas = rasterizeMask(obj.mask, this.size.w, this.size.h);
    s.maskTex.dispose();
    s.maskTex = canvasTexture(s.maskCanvas);

    const H = surfaceHomography(obj.quad, obj.realSize.w, obj.realSize.h, this._lens());
    s.hInv = H ? invert3(H) : null;

    for (const f of FRAMES) {
      const u = s.materials[f].uniforms;
      u.uMask.value = s.maskTex;
      u.uPlaneSize.value.set(obj.realSize.w, obj.realSize.h);
      u.uK1.value = this.k1;
      if (s.hInv) u.uHinv.value.fromArray(s.hInv);
    }
    this._dirty = true;
  }

  addSurface(obj) {
    const px = imagePixels(this.photoTex.image);
    this.surfaces.set(obj.name, this._makeSurface(obj, px));
    this._dirty = true;
  }

  removeSurface(name) {
    const s = this.surfaces.get(name);
    if (s) this._disposeSurface(s);
    this.surfaces.delete(name);
    this._dirty = true;
  }

  // --------------------------------------------------------------- targets --

  _allocTargets() {
    const opts = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
      stencilBuffer: false,
      colorSpace: THREE.LinearSRGBColorSpace,
    };
    const { w, h } = this.size;

    for (const t of this.lumTargets) t.dispose();
    this.lumTargets = [
      new THREE.WebGLRenderTarget(w, h, opts),
      new THREE.WebGLRenderTarget(w, h, opts),
    ];

    for (const f of FRAMES) {
      this.frameTargets[f]?.dispose();
      this.frameTargets[f] = new THREE.WebGLRenderTarget(w, h, {
        ...opts,
        type: THREE.UnsignedByteType,
      });
    }
  }

  /**
   * Build the lighting plate: a two-pass Gaussian over the photo's luminance.
   * The blur radius is what separates "shadow" from "the old floor's own
   * pattern" -- large enough to erase the previous tile grid, small enough to
   * keep the shadow under a chair sharp.
   */
  _buildLightingPlate() {
    const [a, b] = this.lumTargets;
    const u = this.blurMat.uniforms;
    u.uTexel.value.set(1 / this.size.w, 1 / this.size.h);
    u.uRadius.value = this.blurRadius;

    u.uSrc.value = this.photoTex;
    u.uDir.value.set(1, 0);
    u.uPass.value = 0;
    this._draw(this.blurMesh, a);

    u.uSrc.value = a.texture;
    u.uDir.value.set(0, 1);
    u.uPass.value = 1;
    this._draw(this.blurMesh, b);

    for (const s of this.surfaces.values()) {
      for (const f of FRAMES) s.materials[f].uniforms.uBlur.value = b.texture;
    }
  }

  setBlurRadius(px) {
    this.blurRadius = px;
    if (this.photoTex) this._buildLightingPlate();
    this._dirty = true;
  }

  _lens() {
    return { k1: this.k1, width: this.size.w, height: this.size.h };
  }

  /**
   * Change the lens correction. Every homography was solved against the old
   * coefficient, so they all have to be re-solved -- the quads are stored in
   * the photo's own distorted pixels, which is the only space an author can
   * click in.
   */
  setLens(k1) {
    this.k1 = Number(k1) || 0;
    for (const s of this.surfaces.values()) {
      const H = surfaceHomography(s.obj.quad, s.obj.realSize.w, s.obj.realSize.h, this._lens());
      s.hInv = H ? invert3(H) : null;
      for (const f of FRAMES) {
        const u = s.materials[f].uniforms;
        u.uK1.value = this.k1;
        if (s.hInv) u.uHinv.value.fromArray(s.hInv);
      }
    }
    this._dirty = true;
  }

  // -------------------------------------------------------------- products --

  /** Register a product's faces as a tile atlas, once, then reuse it. */
  async loadProduct(product) {
    if (this.products.has(product.id)) return this.products.get(product.id);
    const urls = product.faces?.length ? product.faces : [product.image];
    const atlas = await buildTileAtlas(urls);
    atlas.texture.anisotropy = this.maxAniso;
    const entry = { ...atlas, product };
    this.products.set(product.id, entry);
    return entry;
  }

  // ----------------------------------------------------------------- state --

  /** Push one surface's UI state into its shader uniforms. */
  applyState(frame, surfaceName, state, product) {
    const s = this.surfaces.get(surfaceName);
    if (!s) return;
    const u = s.materials[frame].uniforms;

    const model = materialModel(product?.material);
    u.uMaterial.value = materialModelId(product?.material);
    u.uColor.value.fromArray(hexToRgb(state.color));

    // Each frame's material keeps its own texture, so it must be cleared when
    // this frame has no product (or its product has not loaded yet).
    // Otherwise the surface keeps drawing whatever product it had before, and
    // after a compare swap, keep or reset one side shows the other's tiles
    // instead of the original photo.
    const entry = product ? this.products.get(product.id) : null;
    u.uTile.value = entry ? entry.texture : null;
    if (entry) {
      // Only a modular material picks a different face per unit; a sheet, a
      // rug or a coat of paint is one continuous thing.
      u.uFaces.value = model === 'module' && state.randomFace ? entry.faces : 1;
    }

    u.uTileSize.value.set(state.tileSize.w / 1000, state.tileSize.h / 1000);
    u.uLayout.value = LAYOUT_BY_KEY[state.layout]?.id ?? 0;
    u.uRotation.value = (state.rotation * Math.PI) / 180;
    u.uOffset.value.set(state.offset?.x ?? 0, state.offset?.y ?? 0);
    u.uGrout.value = Math.max(0, (state.grout?.size ?? 0) / 1000);
    u.uGroutColor.value.fromArray(hexToRgb(state.grout?.color));
    u.uTint.value.fromArray(hexToRgb(state.tint));
    u.uBevel.value = state.bevel ?? 0.35;
    u.uGloss.value = state.gloss ?? 0.25;
    u.uShade.value = state.shade ?? 1;
    u.uDetail.value = state.detail ?? 0.6;
    u.uRandomFace.value = state.randomFace ? 1 : 0;
    u.uRandomRotate.value = state.randomRotate ? 1 : 0;
    u.uOpacity.value = state.visible === false ? 0 : 1;

    this.state[frame][surfaceName] = state;
    this._dirty = true;
  }

  setCompare(enabled, split = 0.5) {
    this.compare = { enabled, split };
    this._dirty = true;
  }

  setView({ zoom, x, y }) {
    if (zoom !== undefined) this.view.zoom = zoom;
    if (x !== undefined) this.view.x = x;
    if (y !== undefined) this.view.y = y;
    this._dirty = true;
  }

  setHighlight(name, strength = 0.32) {
    this.highlight = { name, strength: name ? strength : 0 };
    this._dirty = true;
  }

  // ---------------------------------------------------------------- render --

  _draw(mesh, target) {
    this.renderer.setRenderTarget(target ?? null);
    this.renderer.autoClear = true;
    this.scene.clear();
    this.scene.add(mesh);
    this.renderer.render(this.scene, this.camera);
    this.scene.clear();
  }

  _renderFrame(frame) {
    const target = this.frameTargets[frame];
    this.renderer.setRenderTarget(target);
    this.renderer.clear();
    this.scene.clear();
    this.scene.add(this.bgMesh);
    this.renderer.render(this.scene, this.camera);

    this.renderer.autoClear = false;
    for (const s of this.surfaces.values()) {
      const st = this.state[frame][s.obj.name];
      if (!st || st.visible === false || !s.hInv) continue;
      // Paint needs no texture at all, so an absent atlas is not a reason to
      // skip the surface for it.
      if (!s.materials[frame].uniforms.uTile.value
          && s.materials[frame].uniforms.uMaterial.value !== 2) continue;
      this.scene.clear();
      this.scene.add(s.meshes[frame]);
      this.renderer.render(this.scene, this.camera);
    }
    this.renderer.autoClear = true;
    this.scene.clear();
  }

  render() {
    if (!this.room || !this.photoTex) return;

    this._renderFrame('left');
    if (this.compare.enabled) this._renderFrame('right');

    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.round(rect.width * dpr));
    const ch = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.renderer.setSize(cw, ch, false);
    }

    // Aspect-fit the photo inside the canvas: scale is the fraction of the
    // canvas each photo axis occupies.
    const canvasAspect = cw / ch;
    const photoAspect = this.size.w / this.size.h;
    const u = this.presentMat.uniforms;
    if (photoAspect > canvasAspect) u.uScale.value.set(1, canvasAspect / photoAspect);
    else u.uScale.value.set(photoAspect / canvasAspect, 1);

    u.uLeft.value = this.frameTargets.left.texture;
    u.uRight.value = this.frameTargets.right.texture;
    u.uCompare.value = this.compare.enabled ? 1 : 0;
    u.uSplit.value = this.compare.split;
    u.uZoom.value = this.view.zoom;
    u.uPan.value.set(this.view.x, this.view.y);
    u.uCanvas.value.set(cw, ch);

    const hl = this.surfaces.get(this.highlight.name);
    u.uMaskHi.value = hl?.maskTex ?? null;
    u.uHighlight.value = hl ? this.highlight.strength : 0;

    this._draw(this.presentMesh, null);
    this._dirty = false;
  }

  /** Render-on-demand loop: cheap when nothing is moving. */
  start() {
    const tick = () => {
      this._raf = requestAnimationFrame(tick);
      if (this._dirty) this.render();
    };
    if (!this._raf) tick();
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  invalidate() { this._dirty = true; }

  /**
   * Full-resolution export. Reads the composited frame target rather than the
   * on-screen canvas, so the download is the photo's native resolution however
   * the viewport happens to be sized or zoomed.
   */
  exportFrame(frame = 'left', { watermark } = {}) {
    this._renderFrame(frame);
    const { w, h } = this.size;
    const buf = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(this.frameTargets[frame], 0, 0, w, h, buf);

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    // WebGL reads bottom-up; flip into image order.
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w * 4;
      img.data.set(buf.subarray(src, src + w * 4), y * w * 4);
    }
    ctx.putImageData(img, 0, 0);

    if (watermark) {
      ctx.globalAlpha = 0.85;
      const pad = Math.round(w * 0.02);
      const fs = Math.max(14, Math.round(w * 0.022));
      ctx.font = `600 ${fs}px system-ui, sans-serif`;
      ctx.textBaseline = 'bottom';
      const tw = ctx.measureText(watermark).width;
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      ctx.fillRect(pad - 8, h - pad - fs - 8, tw + 16, fs + 14);
      ctx.fillStyle = '#fff';
      ctx.fillText(watermark, pad, h - pad);
    }
    return canvas;
  }

  dispose() {
    this.stop();
    for (const s of this.surfaces.values()) this._disposeSurface(s);
    this.surfaces.clear();
    for (const t of this.lumTargets) t.dispose();
    for (const f of FRAMES) this.frameTargets[f]?.dispose();
    for (const p of this.products.values()) p.texture.dispose();
    this.photoTex?.dispose();
    this.quad.dispose();
    this.renderer.dispose();
  }
}

function scaleMask(mask, scale) {
  if (scale === 1 || !mask) return mask;
  return {
    ...mask,
    polygons: (mask.polygons ?? []).map((p) => ({
      ...p,
      points: p.points.map(([x, y]) => [x * scale, y * scale]),
    })),
  };
}
