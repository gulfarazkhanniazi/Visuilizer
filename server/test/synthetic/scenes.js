/**
 * The synthetic test set: one scene per case the scanner must handle.
 * Camera at the origin, 1.4 m up, looking along +Z unless a case says
 * otherwise. Units are metres.
 */
import { CLS } from '../../src/scanner/pointcloud.js';

const L = { id: 'left', a: [-2, -1], b: [-2, 5], albedo: 0.78 };
const B = { id: 'back', a: [-2, 5], b: [2.5, 5], albedo: 0.82 };
const R = { id: 'right', a: [2.5, 5], b: [2.5, -1], albedo: 0.74 };

export const SCENES = {
  // 1. Two perpendicular walls.
  two_walls: {
    spec: { yaw: -22, hfov: 70, walls: [L, B] },
    expect: { walls: 2, corners: 1 },
  },
  // 2. Left, back and right.
  three_walls: {
    spec: { hfov: 80, walls: [L, B, R] },
    expect: { walls: 3, corners: 2 },
  },
  // 3. Pitched and yawed camera: every wall boundary is diagonal in the image.
  diagonal: {
    spec: { hfov: 75, pitch: -12, yaw: 28, walls: [L, B, R] },
    expect: { walls: 3, corners: 2 },
  },
  // 4. A sofa hides the back wall's skirting.
  furniture: {
    spec: {
      hfov: 80, walls: [L, B, R],
      boxes: [{ min: [-1.2, 0, 4.05], max: [1.3, 0.85, 4.95], albedo: 0.3, cls: CLS.FLOOR_OBJECT }],
    },
    expect: { walls: 3, corners: 2, occluded: 'back' },
  },
  // 5. A floor-to-ceiling wardrobe hides the left/back corner entirely.
  hidden_corner: {
    spec: {
      hfov: 80, walls: [L, B, R],
      boxes: [{ min: [-2, 0, 4.1], max: [-1.1, 2.7, 5], albedo: 0.45, cls: CLS.FLOOR_OBJECT }],
    },
    expect: { walls: 3, corners: 2, inferred: 1 },
  },
  // 6. A window: opening in the wall, glass set back, bright view, frame.
  window: {
    spec: {
      hfov: 80, walls: [L, B, R],
      decos: [
        { wall: 'back', s: [1.4, 3.0], y: [0.9, 2.2], hole: true },
        { wall: 'back', s: [1.4, 3.0], y: [0.9, 2.2], offset: -0.12, emissive: 1.0, cls: CLS.WALL_OBJECT },
        { wall: 'back', s: [1.3, 1.4], y: [0.8, 2.3], offset: 0.03, albedo: 0.15, cls: CLS.WALL_OBJECT },
        { wall: 'back', s: [3.0, 3.1], y: [0.8, 2.3], offset: 0.03, albedo: 0.15, cls: CLS.WALL_OBJECT },
      ],
    },
    expect: { walls: 3, corners: 2 },
  },
  // 7. A door in the right wall: opening, recessed leaf, frame.
  door_frame: {
    spec: {
      hfov: 80, walls: [L, B, R],
      decos: [
        { wall: 'right', s: [1.2, 2.1], y: [0, 2.05], hole: true },
        { wall: 'right', s: [1.2, 2.1], y: [0, 2.05], offset: -0.08, albedo: 0.2, cls: CLS.WALL_OBJECT },
        { wall: 'right', s: [1.1, 1.2], y: [0, 2.15], offset: 0.03, albedo: 0.95, cls: CLS.WALL_OBJECT },
        { wall: 'right', s: [2.1, 2.2], y: [0, 2.15], offset: 0.03, albedo: 0.95, cls: CLS.WALL_OBJECT },
      ],
    },
    expect: { walls: 3, corners: 2 },
  },
  // 8. A dark painting on the back wall.
  painting: {
    spec: {
      hfov: 80, walls: [L, B, R],
      decos: [{ wall: 'back', s: [1.6, 2.9], y: [1.1, 1.9], offset: 0.03, albedo: 0.08, cls: CLS.WALL_OBJECT }],
    },
    expect: { walls: 3, corners: 2 },
  },
  // 9. A hard diagonal shadow across the back wall -- RGB only.
  shadow: {
    spec: {
      hfov: 80, walls: [L, B, R],
      shadows: [{ wall: 'back', s: [1.0, 2.6], y: [0, 2.7], factor: 0.35, slant: 0.5 }],
    },
    expect: { walls: 3, corners: 2 },
  },
  // 10. Weak corner contrast: flat lighting, identical albedo -- the
  // corners are invisible in the luminance image.
  weak_contrast: {
    spec: {
      hfov: 80, light: 'flat',
      walls: [{ ...L, albedo: 0.8 }, { ...B, albedo: 0.8 }, { ...R, albedo: 0.8 }],
    },
    expect: { walls: 3, corners: 2 },
  },
  // 11. Same colour, ordinary lighting.
  same_color: {
    spec: { hfov: 80, walls: [{ ...L, albedo: 0.8 }, { ...B, albedo: 0.8 }, { ...R, albedo: 0.8 }] },
    expect: { walls: 3, corners: 2 },
  },
  // 12. A 135 degree corner (back wall, then a wall angled towards the camera).
  non_90: {
    spec: {
      hfov: 80,
      walls: [L, { id: 'back', a: [-2, 5], b: [1.2, 5], albedo: 0.82 }, { id: 'angled', a: [1.2, 5], b: [3.7, 2.5], albedo: 0.7 }],
    },
    expect: { walls: 3, corners: 2, angles: [90, 135] },
  },
  // 13. A large bright window: most of the back wall is glass.
  bright_window: {
    spec: {
      hfov: 80, walls: [L, B, R],
      decos: [
        { wall: 'back', s: [0.6, 3.9], y: [0.5, 2.4], hole: true },
        { wall: 'back', s: [0.6, 3.9], y: [0.5, 2.4], offset: -0.1, emissive: 1.0, cls: CLS.WALL_OBJECT },
      ],
    },
    expect: { walls: 3, corners: 2 },
  },
  // 14. A dark room: luminance at 8%.
  dark_room: {
    spec: { hfov: 80, brightness: 0.08, walls: [L, B, R] },
    expect: { walls: 3, corners: 2 },
  },
  // Chimney breast: two back-wall pieces either side of a front 40 cm proud.
  chimney: {
    spec: {
      hfov: 80,
      walls: [
        L,
        { id: 'back_l', a: [-2, 5], b: [-0.6, 5], albedo: 0.82 },
        { id: 'side_l', a: [-0.6, 5], b: [-0.6, 4.6], albedo: 0.82 },
        { id: 'front', a: [-0.6, 4.6], b: [0.6, 4.6], albedo: 0.82 },
        { id: 'side_r', a: [0.6, 4.6], b: [0.6, 5], albedo: 0.82 },
        { id: 'back_r', a: [0.6, 5], b: [2.5, 5], albedo: 0.82 },
        R,
      ],
      stepCorners: [{ walls: ['back_l', 'front'], xz: [-0.6, 4.6] }, { walls: ['front', 'back_r'], xz: [0.6, 4.6] }],
    },
    expect: { walls: 5, corners: 4 },
  },
  // Camera tilted down 10 degrees: gravity must come from the floor.
  pitched: {
    spec: { hfov: 75, pitch: -10, walls: [L, B, R] },
    expect: { walls: 3, corners: 2, pitch: -10 },
  },
  // Depth 40% too deep: the room-height prior must recover the scale.
  wrong_scale: {
    spec: { hfov: 80, walls: [L, B, R] },
    corrupt: { scale: 1.4 },
    expect: { walls: 3, corners: 2, camHeight: 1.4 },
  },
};
