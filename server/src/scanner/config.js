/**
 * Every tunable of the 3D scanner in one place.
 *
 * Distances are in metres, angles in degrees, image quantities as fractions of
 * the image width -- never raw pixels -- so the same numbers hold for a 600px
 * phone thumbnail and a 2400px upload. The values were set against the
 * synthetic scenes in test/scanner.*.test.js (which measure what each one
 * does) and checked on the real photographs in test-images/.
 */
export const GEOMETRY_VERSION = 'geometry3d-1.0.0';

export const QUALITY = {
  // Geometry grid width: where point cloud, normals and planes live. Final
  // masks are always assigned at the refinement resolution, not this one.
  fast: { gridW: 256, lines: false, sam: false, corners: 'geometry', focalSearch: false },
  balanced: { gridW: 384, lines: true, sam: false, corners: 'full', focalSearch: true },
  high: { gridW: 512, lines: true, sam: true, corners: 'full', focalSearch: true },
};

export const CAMERA = {
  defaultHfov: 70,           // used only when nothing better is known
  hfovRange: [38, 105],      // accepted range for an estimated field of view
  levelSnapDeg: 1.0,         // pitch/roll below this is treated as level
  plausibleCamHeight: [1.0, 1.9], // metric depth is trusted when it puts the camera here
};

export const DEPTH = {
  maxRelGradient: 0.12,      // |grad Z| / Z above this is a silhouette, not surface
  minDepth: 0.2,
  maxDepth: 30,
};

export const PLANES = {
  // Point-to-plane inlier tolerance grows with distance, because monocular
  // depth error does: tol = abs + rel * Z.
  tolAbs: 0.03,
  tolRel: 0.035,
  normalTolDeg: 28,          // point normal vs plane normal for membership
  maxWallTiltDeg: 20,        // a seed point's normal must be this close to horizontal
  angleBinDeg: 3,
  angleSmoothBins: 2,
  minPeakFrac: 0.035,        // a direction peak must hold this share of wall points
  minPeakSepDeg: 14,
  // Offsets are histogrammed in log(distance), so bins are relative.
  offsetBinLog: 0.012,
  minOffsetSepLog: 0.035,    // parallel planes 3.5% of distance apart can separate
  minOffsetPeakFrac: 0.08,
  offsetSmoothBins: 1,
  ransacIters: 180,
  minWallPointsFrac: 0.004,  // of grid pixels
  minComponentFrac: 0.0025,
  // Validation: a plane failing these is rejected rather than shipped.
  maxMedianRelResidual: 0.035,
  maxP95RelResidual: 0.12,
  minNormalConsistency: 0.72,
  minInlierRatio: 0.45,
  // Merge rule: two regions become one wall only when ALL of these hold and
  // no validated corner separates them.
  mergeAngleDeg: 7,
  // ... and one plane fits the union nearly as well as two fit the halves.
  mergeResidualGain: 1.6,
  mergeResidualFloor: 0.006,
  // Profile split (per wall, along image columns): split only where two lines
  // explain the profile far better than one AND the pieces physically differ.
  splitRelRms: 0.006,
  splitGain: 0.35,
  splitMinColsFrac: 0.03,
  splitMinBendDeg: 12,
  splitMinStep: 0.12,
  splitMinStepRel: 0.025,
  // Two coplanar pieces separated by another wall plane are separate walls
  // (a chimney breast between them); separated only by furniture, one wall.
  maxForeignGapFrac: 0.3,
};

export const CORNERS = {
  minAngleDeg: 12,           // planes closer to parallel than this cannot meet
  maxExtension: 1.6,         // m a wall may be extended behind an occluder to reach its corner
  maxExtensionFrac: 0.6,     // ... or this fraction of its visible width, whichever is larger
  insideTolerance: 0.3,      // m the intersection may sit inside a visible extent
  band: 0.018,               // evidence band either side of the line, fraction of width
  depthBandInner: 0.012,     // depth/normal evidence skips this much either side...
  depthBandOuter: 0.05,      // ...and reads the surfaces out to this distance
  boundarySigma: 0.012,      // plane-intersection vs observed boundary, fraction of width
  samples: 48,
  // Evidence weights. Geometry dominates by design: an RGB edge alone can
  // never reach the acceptance threshold.
  weights: {
    planeIntersection: 0.30,
    depth: 0.20,
    normals: 0.20,
    lines: 0.10,
    roomGeometry: 0.10,
    segmentation: 0.05,
    rgb: 0.05,
    // Only scored for a corner nobody can see: is something standing in
    // front of it? (Otherwise it would have been observed.)
    occlusion: 0.25,
  },
  accept: 0.5,
  // An observed corner additionally needs this much depth or normal evidence.
  minGeometricSupport: 0.35,
  // Share of the line that must be covered by objects for the corner to be
  // reported as inferred rather than observed.
  occludedFrac: 0.6,
  // Adjacent walls whose corner candidate is rejected on evidence, and whose
  // planes are closer to parallel than this, are one wall split by noise.
  mergeWithoutCornerDeg: 20,
  // A step between parallel planes is a vertical line with one wall on each
  // side. When more than this share of the two walls' pixels (in the rows
  // they share) lies on the wrong side of it, they are stacked one above the
  // other -- one wall split horizontally by depth noise -- not a step.
  stepMaxWrongSide: 0.4,
};

export const ASSIGN = {
  depthTolRel: 0.18,         // ray hit vs observed depth
  depthTolRelLoose: 0.35,
  // A wall-labelled pixel below eye level whose observed depth is this far in
  // front of the plane is furniture the segmenter missed (a sofa back).
  frontTolRel: 0.12,
  labelBonus: 0.03,          // preference for the plane the fit assigned the pixel to
  tieRel: 0.004,             // depth agreements this close are a tie -> nearer hit
  flushTolRel: 0.015,        // an object this close to a wall's plane lies flush on it
  extentPad: 0.12,           // m slack on a wall's extent when testing a hit
  heightPad: 0.25,
  minWallZ: 0.3,             // m: a quad corner nearer than this is clipped
};

/** Confidence bands. Reported with every scan so the UI can show them. */
export const CONFIDENCE_BANDS = [
  { min: 0.9, label: 'high' },
  { min: 0.75, label: 'good' },
  { min: 0.55, label: 'uncertain' },
  { min: 0, label: 'unreliable' },
];

export function confidenceLabel(c) {
  return CONFIDENCE_BANDS.find((b) => c >= b.min)?.label ?? 'unreliable';
}
