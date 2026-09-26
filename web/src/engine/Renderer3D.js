import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { meshVertexShader, meshFragmentShader } from './meshShader.js';
import { buildTileAtlas, hexToRgb } from './textures.js';
import { LAYOUT_BY_KEY, materialModelId, materialModel } from './layouts.js';

const FRAMES = ['left', 'right'];

/**
 * The 3D room renderer.
 *
 * A modelled room is the one case where the geometry is simply given: no
 * homography to solve, no ray to cast, no mask to draw. What it costs instead
 * is a camera the visitor drives, so this is the only renderer with real orbit
 * controls and a perspective camera rather than a full-screen quad.
 *
 * Which meshes are surfaces is decided by the room's objectList: each entry
 * names a mesh in the glTF file. Everything the room's author did not tag --
 * furniture, joinery, the kettle -- keeps the material it shipped with, which
 * is what makes the tiled surfaces sit in a believable room.
 */
export default class Renderer3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    // Unlike the photo renderers, this one is lighting a scene rather than
    // compositing onto a photograph, so three's colour management is what we
    // want rather than what we have to switch off.
    THREE.ColorManagement.enabled = true;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x0f1115, 1);
    this.maxAniso = this.renderer.capabilities.getMaxAnisotropy();

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(52, 1, 0.05, 200);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.55;
    this.controls.panSpeed = 0.6;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 40;
    // Stop the orbit dropping below the floor, which is the single easiest way
    // to end up looking at the underside of a room and not knowing why.
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener('change', () => { this._dirty = true; });

    this.key = new THREE.DirectionalLight(0xffffff, 2.1);
    this.key.position.set(3.2, 5.4, 2.6);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(1024, 1024);
    this.key.shadow.bias = -0.0012;
    this.scene.add(this.key);
    this.scene.add(new THREE.HemisphereLight(0xdfe8f5, 0x40372e, 1.15));

    this.model = null;
    this.room = null;
    this.surfaces = new Map();     // surface name -> { obj, meshes, materials }
    this.products = new Map();
    this.state = { left: {}, right: {} };
    this.frame = 'left';
    this.home = null;

    this._dirty = true;
    this._raf = null;
  }

  // ------------------------------------------------------------------ room --

  async setRoom(room, modelUrl) {
    this.room = room;

    if (this.model) {
      this.scene.remove(this.model);
      disposeTree(this.model);
      this.model = null;
    }
    this.surfaces.clear();

    const gltf = await new GLTFLoader().loadAsync(modelUrl);
    this.model = gltf.scene;

    // Drop the model onto the origin and scale nothing: the tiling reads world
    // metres, so a model authored in centimetres would lay 60 mm tiles. That
    // is the author's problem to fix in the exporter, not something to paper
    // over here -- but centre it, so the orbit target is sane.
    const box = new THREE.Box3().setFromObject(this.model);
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    this.model.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.receiveShadow = true;
    });
    this.scene.add(this.model);

    const byName = new Map();
    this.model.traverse((o) => { if (o.isMesh) byName.set(o.name, o); });

    for (const obj of room.objectList ?? []) {
      const names = obj.meshes?.length ? obj.meshes : [obj.name];
      const meshes = names.map((n) => byName.get(n)).filter(Boolean);
      if (!meshes.length) continue;

      const bounds = new THREE.Box3();
      for (const m of meshes) bounds.expandByObject(m);
      const span = bounds.getSize(new THREE.Vector3());

      const materials = {};
      for (const f of FRAMES) {
        const mat = this._newSurfaceMaterial();
        // The plane extent a rug is centred on, in the two axes the box
        // mapping actually uses for this surface.
        const horizontal = span.y <= Math.min(span.x, span.z);
        mat.uniforms.uPlaneSize.value.set(
          horizontal ? span.x : Math.max(span.x, span.z),
          horizontal ? span.z : span.y,
        );
        materials[f] = mat;
      }

      // Every mesh in the group renders with the active frame's material, so
      // switching frames is a material swap rather than a rebuild.
      for (const m of meshes) m.material = materials.left;

      this.surfaces.set(obj.name, { obj, meshes, materials, original: meshes.map((m) => m.material) });
    }

    // Frame the room: back off far enough for its largest dimension to fit.
    const radius = Math.max(size.x, size.y, size.z) * 0.62;
    const dist = radius / Math.tan((this.camera.fov * Math.PI) / 360);
    const eye = new THREE.Vector3(centre.x + dist * 0.55, centre.y + size.y * 0.22, centre.z + dist * 0.75);
    this.home = { eye: eye.clone(), target: centre.clone() };
    this.resetView();

    this.key.target.position.copy(centre);
    this.scene.add(this.key.target);

    this._dirty = true;
    return this;
  }

  _newSurfaceMaterial() {
    return new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: meshVertexShader,
      fragmentShader: meshFragmentShader,
      transparent: true,
      uniforms: {
        uLightDir: { value: new THREE.Vector3(0.42, 0.78, 0.46).normalize() },
        uAmbient: { value: 0.34 },
        uOriginOffset: { value: new THREE.Vector3() },
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
        uRefLevel: { value: 0.78 },
        uShade: { value: 1 },
        uDetail: { value: 0 },
        uGloss: { value: 0.25 },
        uOpacity: { value: 1 },
      },
    });
  }

  // -------------------------------------------------------------- products --

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
    const s = this.surfaces.get(surfaceName);
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
    u.uRandomFace.value = state.randomFace ? 1 : 0;
    u.uRandomRotate.value = state.randomRotate ? 1 : 0;

    // A hidden surface falls back to whatever the model shipped with, so the
    // room still reads as a room rather than as a hole.
    const hide = state.visible === false;
    s.meshes.forEach((m, i) => {
      m.material = hide ? s.original[i] : s.materials[frame];
    });

    this.state[frame][surfaceName] = state;
    this._dirty = true;
  }

  /** Which of the two comparison looks is on screen. */
  setFrame(frame) {
    if (frame === this.frame) return;
    this.frame = frame;
    for (const s of this.surfaces.values()) {
      const st = this.state[frame][s.obj.name];
      if (st?.visible === false) continue;
      for (const m of s.meshes) m.material = s.materials[frame];
    }
    this._dirty = true;
  }

  // ---------------------------------------------------------------- camera --

  resetView() {
    if (!this.home) return;
    this.camera.position.copy(this.home.eye);
    this.controls.target.copy(this.home.target);
    this.controls.update();
    this._dirty = true;
  }

  setAutoRotate(on) {
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = 0.7;
    this._dirty = true;
  }

  zoom(delta) {
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    const len = THREE.MathUtils.clamp(dir.length() * (1 + delta), this.controls.minDistance, this.controls.maxDistance);
    this.camera.position.copy(this.controls.target).add(dir.setLength(len));
    this.controls.update();
    this._dirty = true;
  }

  // ---------------------------------------------------------------- render --

  render() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
    this._dirty = false;
  }

  start() {
    const tick = () => {
      this._raf = requestAnimationFrame(tick);
      // Damping and auto-rotate both keep moving after the pointer stops, so
      // the controls decide when the frame is stale rather than a flag alone.
      const moving = this.controls.enableDamping || this.controls.autoRotate;
      if (moving) this.controls.update();
      if (this._dirty || this.controls.autoRotate) this.render();
    };
    if (!this._raf) tick();
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  invalidate() { this._dirty = true; }

  /** Snapshot of the current view, for downloads, shares and PDF sheets. */
  exportView(scale = 2) {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(640, Math.round(rect.width * scale));
    const h = Math.max(420, Math.round(rect.height * scale));

    const target = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      type: THREE.UnsignedByteType,
      colorSpace: THREE.SRGBColorSpace,
    });
    const aspect = this.camera.aspect;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, this.camera);
    const buf = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(target, 0, 0, w, h, buf);
    this.renderer.setRenderTarget(null);

    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    target.dispose();

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
    this._dirty = true;
    return canvas;
  }

  dispose() {
    this.stop();
    this.controls.dispose();
    for (const s of this.surfaces.values()) {
      for (const f of FRAMES) s.materials[f].dispose();
    }
    this.surfaces.clear();
    if (this.model) disposeTree(this.model);
    for (const p of this.products.values()) p.texture.dispose();
    this.renderer.dispose();
  }
}

/** Free everything a loaded glTF allocated on the GPU. */
function disposeTree(root) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry?.dispose();
    for (const m of [].concat(o.material ?? [])) {
      for (const k of Object.keys(m)) {
        const v = m[k];
        if (v && v.isTexture) v.dispose();
      }
      m.dispose?.();
    }
  });
}

/**
 * Read the mesh names out of a glTF without building a renderer.
 *
 * The Studio needs this to offer the author a list to tag, and it must not
 * depend on a WebGL context existing -- the page showing the list is a form,
 * not a viewer.
 */
export async function readModelMeshes(url) {
  const gltf = await new GLTFLoader().loadAsync(url);
  const out = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const box = new THREE.Box3().setFromObject(o);
    const size = box.getSize(new THREE.Vector3());
    out.push({
      name: o.name || `mesh_${out.length}`,
      triangles: (o.geometry?.index?.count ?? o.geometry?.attributes?.position?.count ?? 0) / 3,
      size: { x: round2(size.x), y: round2(size.y), z: round2(size.z) },
      // A guess at what it is, so a room with sane mesh names is one click to
      // set up rather than twenty.
      guess: guessSurface(o.name, size),
    });
  });
  disposeTree(gltf.scene);
  return out;
}

const round2 = (n) => Math.round(n * 100) / 100;

function guessSurface(name, size) {
  const n = (name ?? '').toLowerCase();
  if (/floor|ground|sol|boden/.test(n)) return 'floor';
  if (/ceil|plafond|decke/.test(n)) return 'ceiling';
  if (/splash|backsplash/.test(n)) return 'backsplash';
  if (/counter|worktop/.test(n)) return 'countertop';
  if (/wall|mur|wand/.test(n)) return 'wall';
  // Nothing in the name: fall back to shape. A big flat thing lying down is a
  // floor; a big flat thing standing up is a wall.
  const flatDown = size.y < Math.min(size.x, size.z) * 0.2;
  const flatUp = Math.min(size.x, size.z) < size.y * 0.35;
  if (flatDown && size.x * size.z > 1.5) return 'floor';
  if (flatUp && size.y > 1.4) return 'wall';
  return '';
}
