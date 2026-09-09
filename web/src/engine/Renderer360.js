import * as THREE from 'three';
import {
  fullscreenVertexShader,
  panoFragmentShader,
  blurFragmentShader,
} from './surfaceShader.js';
import { loadImage, buildTileAtlas, imageTexture, imagePixels, hexToRgb } from './textures.js';
import { LAYOUT_BY_KEY, materialModelId, materialModel } from './layouts.js';


const MAX_PLANES = 8;
const FRAMES = ['left', 'right'];

/**
 * The 360 renderer.
 *
 * Compositing happens in equirectangular space, not on screen: each texel of
 * the panorama is a direction, so casting that ray at the room's surface planes
 * and keeping the nearest hit gives both the surface and its metric coordinate.
 * Occlusion falls out of the geometry, so a 360 room needs no masks at all.
 *
 * The composited panorama is then just a texture on a sphere, which makes the
 * viewer a plain look-around camera with optional gyroscope.
 */
export default class Renderer360 {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    // Same reason as the flat renderer: the relighting maths is done on
    // sRGB-encoded values, so three must not re-encode them.
    THREE.ColorManagement.enabled = false;
    this.renderer.setClearColor(0x0d0f13, 1);
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.maxAniso = this.renderer.capabilities.getMaxAnisotropy();

    // --- offscreen compositing ---
    this.quadScene = new THREE.Scene();
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.PlaneGeometry(2, 2);

    // --- the sphere we actually look at ---
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    this.camera.position.set(0, 0, 0);
    const sphere = new THREE.SphereGeometry(50, 72, 48);
    sphere.scale(-1, 1, 1);          // view it from the inside
    this.sphereMat = new THREE.MeshBasicMaterial({ toneMapped: false });
    this.sphere = new THREE.Mesh(sphere, this.sphereMat);
    this.scene.add(this.sphere);

    this.room = null;
    this.panoTex = null;
    this.size = { w: 1, h: 1 };
    this.surfaces = [];
    this.products = new Map();
    this.state = { left: {}, right: {} };
    this.view = { yaw: 0, pitch: -0.22, fov: 75 };
    this.gyro = false;
    this.autoRotate = false;
    this.blurRadius = 9;

    this._buildPasses();
    this._dirty = true;
    this._raf = null;
    this._lastTime = 0;
  }

  _buildPasses() {
    this.blurMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: blurFragmentShader,
      uniforms: {
        uSrc: { value: null },
        uTexel: { value: new THREE.Vector2() },
        uDir: { value: new THREE.Vector2(1, 0) },
        uRadius: { value: 9 },
        uPass: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.blurMesh = new THREE.Mesh(this.quad, this.blurMat);

    this.bgMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: /* glsl */ `
        precision highp float;
        in vec2 vUv;
        out vec4 outColor;
        uniform sampler2D uPano;
        void main() { outColor = vec4(texture(uPano, vUv).rgb, 1.0); }
      `,
      uniforms: { uPano: { value: null } },
      depthTest: false,
      depthWrite: false,
    });
    this.bgMesh = new THREE.Mesh(this.quad, this.bgMat);
  }

  _newSurfaceMaterial() {
    const zeros = () => Array.from({ length: MAX_PLANES }, () => new THREE.Vector3());
    return new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertexShader,
      fragmentShader: panoFragmentShader,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: {
        uPano: { value: null },
        uBlur: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uCount: { value: 0 },
        uActive: { value: 0 },
        uOrigin: { value: zeros() },
        uNormal: { value: zeros() },
        uAxisU: { value: zeros() },
        uAxisV: { value: zeros() },
        uExtent: { value: Array.from({ length: MAX_PLANES }, () => new THREE.Vector4()) },
        uTile: { value: null },
        uFaces: { value: 1 },
        uTileSize: { value: new THREE.Vector2(0.6, 0.6) },
        uTint: { value: new THREE.Vector3(1, 1, 1) },
        uMaterial: { value: 0 },
        uColor: { value: new THREE.Vector3(0.92, 0.89, 0.84) },
        uPlaneSize: { value: new THREE.Vector2(3, 3) },
        uLayout: { value: 0 },
        uRotation: { value: 0 },
        uOffset: { value: new THREE.Vector2() },
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

  async setRoom(room, panoUrl) {
    const img = await loadImage(panoUrl);
    this.room = room;
    this.size = { w: img.naturalWidth, h: img.naturalHeight };

    this.panoTex?.dispose();
    this.panoTex = imageTexture(img);
    // The panorama wraps horizontally; without this the seam shows as a stripe.
    this.panoTex.wrapS = THREE.RepeatWrapping;
    this.bgMat.uniforms.uPano.value = this.panoTex;

    this._allocTargets();
    this._buildLightingPlate();

    // One reference luminance per surface, sampled over the whole panorama --
    // the planes tell us which texels belong to which surface.
    const px = imagePixels(img, 512);
    const refs = this._referenceLevels(px);

    for (const s of this.surfaces) for (const f of FRAMES) s.materials[f].dispose();
    this.surfaces = (room.objectList ?? []).filter((o) => o.plane).slice(0, MAX_PLANES)
      .map((obj, i) => {
        const materials = {};
        for (const f of FRAMES) {
          const m = this._newSurfaceMaterial();
          m.uniforms.uPano.value = this.panoTex;
          m.uniforms.uBlur.value = this.lumTargets[1].texture;
          m.uniforms.uResolution.value.set(this.size.w, this.size.h);
          m.uniforms.uActive.value = i;
          m.uniforms.uRefLevel.value = refs[obj.name] ?? 0.5;
          // The plane's own span in metres, so a rug can be centred on it.
          const e = obj.plane.extent;
          m.uniforms.uPlaneSize.value.set(Math.abs(e[1] - e[0]), Math.abs(e[3] - e[2]));
          materials[f] = m;
        }
        return {
          obj,
          index: i,
          materials,
          meshes: Object.fromEntries(FRAMES.map((f) => [f, new THREE.Mesh(this.quad, materials[f])])),
        };
      });

    this._pushPlanes();
    this._dirty = true;
    return this;
  }

  /** Copy every plane into every surface material's uniform arrays. */
  _pushPlanes() {
    const n = this.surfaces.length;
    for (const s of this.surfaces) {
      for (const f of FRAMES) {
        const u = s.materials[f].uniforms;
        u.uCount.value = n;
        this.surfaces.forEach((other, j) => {
          const p = other.obj.plane;
          u.uOrigin.value[j].fromArray(p.origin);
          u.uNormal.value[j].fromArray(p.normal).normalize();
          u.uAxisU.value[j].fromArray(p.axisU).normalize();
          u.uAxisV.value[j].fromArray(p.axisV).normalize();
          u.uExtent.value[j].fromArray(p.extent);
        });
      }
    }
  }

  /**
   * Which panorama texels belong to which surface, done on the CPU with the
   * same nearest-hit rule the shader uses, so the reference levels match what
   * gets rendered.
   */
  _referenceLevels(px) {
    const planes = (this.room.objectList ?? []).filter((o) => o.plane);
    const hist = Object.fromEntries(planes.map((p) => [p.name, []]));

    for (let y = 0; y < px.height; y += 2) {
      const lat = (0.5 - (y + 0.5) / px.height) * Math.PI;
      const cl = Math.cos(lat);
      const dy = Math.sin(lat);
      for (let x = 0; x < px.width; x += 2) {
        const lon = ((x + 0.5) / px.width - 0.5) * 2 * Math.PI;
        const d = [cl * Math.sin(lon), dy, -cl * Math.cos(lon)];

        let best = Infinity;
        let hit = null;
        for (const o of planes) {
          const pl = o.plane;
          const denom = dot(pl.normal, d);
          if (Math.abs(denom) < 1e-6) continue;
          const t = dot(pl.normal, pl.origin) / denom;
          if (t <= 1e-4 || t >= best) continue;
          const p = [d[0] * t, d[1] * t, d[2] * t];
          const rel = [p[0] - pl.origin[0], p[1] - pl.origin[1], p[2] - pl.origin[2]];
          const a = dot(rel, pl.axisU);
          const b = dot(rel, pl.axisV);
          const [u0, u1, v0, v1] = pl.extent;
          if (a < u0 || a > u1 || b < v0 || b > v1) continue;
          best = t; hit = o.name;
        }
        if (!hit) continue;
        const i = (y * px.width + x) * 4;
        const l = (0.2126 * px.data[i] + 0.7152 * px.data[i + 1] + 0.0722 * px.data[i + 2]) / 255;
        hist[hit].push(l);
      }
    }

    const out = {};
    for (const [name, list] of Object.entries(hist)) {
      if (!list.length) { out[name] = 0.5; continue; }
      list.sort((a, b) => a - b);
      out[name] = Math.max(0.02, list[Math.floor(list.length * 0.6)]);
    }
    return out;
  }

  _allocTargets() {
    const base = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
      colorSpace: THREE.LinearSRGBColorSpace,
      wrapS: THREE.RepeatWrapping,
    };
    const { w, h } = this.size;

    for (const t of this.lumTargets ?? []) t.dispose();
    this.lumTargets = [
      new THREE.WebGLRenderTarget(w, h, { ...base, type: THREE.HalfFloatType }),
      new THREE.WebGLRenderTarget(w, h, { ...base, type: THREE.HalfFloatType }),
    ];

    for (const f of FRAMES) {
      this.frameTargets ??= {};
      this.frameTargets[f]?.dispose();
      this.frameTargets[f] = new THREE.WebGLRenderTarget(w, h, {
        ...base,
        type: THREE.UnsignedByteType,
        generateMipmaps: true,
        minFilter: THREE.LinearMipmapLinearFilter,
      });
      this.frameTargets[f].texture.anisotropy = this.maxAniso;
    }
  }

  _buildLightingPlate() {
    const [a, b] = this.lumTargets;
    const u = this.blurMat.uniforms;
    u.uTexel.value.set(1 / this.size.w, 1 / this.size.h);
    u.uRadius.value = this.blurRadius;

    u.uSrc.value = this.panoTex;
    u.uDir.value.set(1, 0);
    u.uPass.value = 0;
    this._draw(this.blurMesh, a);

    u.uSrc.value = a.texture;
    u.uDir.value.set(0, 1);
    u.uPass.value = 1;
    this._draw(this.blurMesh, b);

    for (const s of this.surfaces) {
      for (const f of FRAMES) s.materials[f].uniforms.uBlur.value = b.texture;
    }
  }

  setBlurRadius(px) {
    this.blurRadius = px;
    if (this.panoTex) this._buildLightingPlate();
    this._dirty = true;
  }

  async loadProduct(product) {
    if (this.products.has(product.id)) return this.products.get(product.id);
    const urls = product.faces?.length ? product.faces : [product.image];
    const atlas = await buildTileAtlas(urls);
    atlas.texture.anisotropy = this.maxAniso;
    const entry = { ...atlas, product };
    this.products.set(product.id, entry);
    return entry;
  }

  applyState(frame, surfaceName, state, product) {
    const s = this.surfaces.find((x) => x.obj.name === surfaceName);
    if (!s) return;
    const u = s.materials[frame].uniforms;

    const model = materialModel(product?.material);
    u.uMaterial.value = materialModelId(product?.material);
    u.uColor.value.fromArray(hexToRgb(state.color));

    if (product) {
      const entry = this.products.get(product.id);
      if (entry) {
        u.uTile.value = entry.texture;
        u.uFaces.value = model === 'module' && state.randomFace ? entry.faces : 1;
      }
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

  // ---------------------------------------------------------------- camera --

  setView({ yaw, pitch, fov }) {
    if (yaw !== undefined) this.view.yaw = yaw;
    if (pitch !== undefined) {
      // Stop short of the poles, where an equirect panorama smears.
      this.view.pitch = Math.max(-1.45, Math.min(1.45, pitch));
    }
    if (fov !== undefined) this.view.fov = Math.max(28, Math.min(100, fov));
    this._dirty = true;
  }

  look(dx, dy) {
    const k = (this.view.fov / 75) * 0.0032;
    this.setView({ yaw: this.view.yaw - dx * k, pitch: this.view.pitch - dy * k });
  }

  setAutoRotate(on) { this.autoRotate = on; this._dirty = true; }

  /**
   * Phone tilt. Only ever enabled from a user gesture, because iOS requires
   * DeviceOrientationEvent.requestPermission to be called from one.
   */
  async enableGyro() {
    const DOE = window.DeviceOrientationEvent;
    if (!DOE) throw new Error('This device has no orientation sensor.');
    if (typeof DOE.requestPermission === 'function') {
      const res = await DOE.requestPermission();
      if (res !== 'granted') throw new Error('Motion access was denied.');
    }
    this._onOrient = (e) => {
      if (e.alpha == null) return;
      this.setView({
        yaw: THREE.MathUtils.degToRad(e.alpha),
        pitch: THREE.MathUtils.degToRad(Math.max(-80, Math.min(80, e.beta - 90))),
      });
    };
    window.addEventListener('deviceorientation', this._onOrient);
    this.gyro = true;
  }

  disableGyro() {
    if (this._onOrient) window.removeEventListener('deviceorientation', this._onOrient);
    this._onOrient = null;
    this.gyro = false;
  }

  // ---------------------------------------------------------------- render --

  _draw(mesh, target) {
    this.renderer.setRenderTarget(target ?? null);
    this.renderer.autoClear = true;
    this.quadScene.clear();
    this.quadScene.add(mesh);
    this.renderer.render(this.quadScene, this.quadCam);
    this.quadScene.clear();
  }

  _composite(frame) {
    const target = this.frameTargets[frame];
    this.renderer.setRenderTarget(target);
    this.renderer.clear();
    this.quadScene.clear();
    this.quadScene.add(this.bgMesh);
    this.renderer.render(this.quadScene, this.quadCam);

    this.renderer.autoClear = false;
    for (const s of this.surfaces) {
      const st = this.state[frame][s.obj.name];
      if (!st || st.visible === false) continue;
      if (!s.materials[frame].uniforms.uTile.value
          && s.materials[frame].uniforms.uMaterial.value !== 2) continue;
      this.quadScene.clear();
      this.quadScene.add(s.meshes[frame]);
      this.renderer.render(this.quadScene, this.quadCam);
    }
    this.renderer.autoClear = true;
    this.quadScene.clear();
    target.texture.needsUpdate = true;
  }

  render(dt = 0) {
    if (!this.room || !this.panoTex) return;

    if (this._needsComposite !== false) {
      this._composite('left');
      this.sphereMat.map = this.frameTargets.left.texture;
      this.sphereMat.needsUpdate = true;
      this._needsComposite = false;
    }

    if (this.autoRotate && !this.gyro) this.view.yaw += dt * 0.06;

    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.round(rect.width * dpr));
    const ch = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.renderer.setSize(cw, ch, false);
      this.camera.aspect = cw / ch;
    }
    this.camera.fov = this.view.fov;
    this.camera.updateProjectionMatrix();

    const { yaw, pitch } = this.view;
    this.camera.lookAt(
      Math.cos(pitch) * Math.sin(yaw),
      Math.sin(pitch),
      -Math.cos(pitch) * Math.cos(yaw),
    );

    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
  }

  invalidate() { this._dirty = true; this._needsComposite = true; }

  start() {
    const tick = (t) => {
      this._raf = requestAnimationFrame(tick);
      const dt = this._lastTime ? (t - this._lastTime) / 1000 : 0;
      this._lastTime = t;
      if (this._dirty || this.autoRotate || this.gyro) {
        this.render(dt);
        if (!this.autoRotate && !this.gyro) this._dirty = false;
      }
    };
    if (!this._raf) this._raf = requestAnimationFrame(tick);
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  /** Flat equirectangular export of the composited panorama. */
  exportPanorama(frame = 'left') {
    this._composite(frame);
    const { w, h } = this.size;
    const buf = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(this.frameTargets[frame], 0, 0, w, h, buf);

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w * 4;
      img.data.set(buf.subarray(src, src + w * 4), y * w * 4);
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  /** What is on screen right now, for downloads and PDFs. */
  exportView() {
    this.render(0);
    const canvas = document.createElement('canvas');
    canvas.width = this.canvas.width;
    canvas.height = this.canvas.height;
    canvas.getContext('2d').drawImage(this.canvas, 0, 0);
    return canvas;
  }

  dispose() {
    this.stop();
    this.disableGyro();
    for (const s of this.surfaces) for (const f of FRAMES) s.materials[f].dispose();
    for (const t of this.lumTargets ?? []) t.dispose();
    for (const f of FRAMES) this.frameTargets?.[f]?.dispose();
    for (const p of this.products.values()) p.texture.dispose();
    this.panoTex?.dispose();
    this.quad.dispose();
    this.sphere.geometry.dispose();
    this.sphereMat.dispose();
    this.renderer.dispose();
  }
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
