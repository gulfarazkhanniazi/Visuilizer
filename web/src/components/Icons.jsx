/** Inline icons: no icon-font request, and they inherit currentColor. */
const s = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
};

const wrap = (children, props) => (
  <svg viewBox="0 0 24 24" width="1em" height="1em" {...s} {...props}>{children}</svg>
);

export const IconSurfaces = (p) => wrap(<>
  <path d="M3 16.5 12 21l9-4.5" /><path d="M3 12 12 16.5 21 12" /><path d="m3 7.5 9-4.5 9 4.5-9 4.5z" />
</>, p);

export const IconTiles = (p) => wrap(<>
  <rect x="3" y="3" width="7.5" height="7.5" rx="1" />
  <rect x="13.5" y="3" width="7.5" height="7.5" rx="1" />
  <rect x="3" y="13.5" width="7.5" height="7.5" rx="1" />
  <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1" />
</>, p);

export const IconLayout = (p) => wrap(<>
  <rect x="3" y="4" width="18" height="6" rx="1" />
  <rect x="3" y="14" width="8" height="6" rx="1" />
  <rect x="13" y="14" width="8" height="6" rx="1" />
</>, p);

export const IconGrout = (p) => wrap(<>
  <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
  <rect x="3" y="3" width="18" height="18" rx="2" />
</>, p);

export const IconFinish = (p) => wrap(<>
  <circle cx="12" cy="12" r="9" /><path d="M12 3a9 9 0 0 0 0 18" fill="currentColor" stroke="none" opacity=".35" />
</>, p);

export const IconCompare = (p) => wrap(<>
  <rect x="3" y="4" width="18" height="16" rx="2" /><path d="M12 4v16" />
  <path d="M8 12h-3M6.5 10.5 5 12l1.5 1.5M16 12h3M17.5 10.5 19 12l-1.5 1.5" />
</>, p);

export const IconDownload = (p) => wrap(<>
  <path d="M12 3v12" /><path d="m7.5 10.5 4.5 4.5 4.5-4.5" /><path d="M4 20h16" />
</>, p);

export const IconShare = (p) => wrap(<>
  <circle cx="18" cy="5" r="2.5" /><circle cx="6" cy="12" r="2.5" /><circle cx="18" cy="19" r="2.5" />
  <path d="m8.2 10.8 7.6-4.1M8.2 13.2l7.6 4.1" />
</>, p);

export const IconUndo = (p) => wrap(<>
  <path d="M9 8H5V4" /><path d="M5.5 8.5a8 8 0 1 1-.5 7" />
</>, p);

export const IconRedo = (p) => wrap(<>
  <path d="M15 8h4V4" /><path d="M18.5 8.5a8 8 0 1 0 .5 7" />
</>, p);

export const IconZoomIn = (p) => wrap(<>
  <circle cx="11" cy="11" r="7" /><path d="m20 20-3.6-3.6M11 8v6M8 11h6" />
</>, p);

export const IconZoomOut = (p) => wrap(<>
  <circle cx="11" cy="11" r="7" /><path d="m20 20-3.6-3.6M8 11h6" />
</>, p);

export const IconReset = (p) => wrap(<>
  <path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" />
</>, p);

export const IconUpload = (p) => wrap(<>
  <path d="M12 16V4" /><path d="m7.5 8.5 4.5-4.5 4.5 4.5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
</>, p);

export const IconClose = (p) => wrap(<><path d="m6 6 12 12M18 6 6 18" /></>, p);

export const IconCheck = (p) => wrap(<><path d="m5 13 4.5 4.5L19 7" /></>, p);

export const IconTrash = (p) => wrap(<>
  <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />
</>, p);

export const IconPlus = (p) => wrap(<><path d="M12 5v14M5 12h14" /></>, p);

export const IconEdit = (p) => wrap(<>
  <path d="M4 20h4L19 9a2.1 2.1 0 0 0-3-3L5 17z" /><path d="M14.5 6.5 17.5 9.5" />
</>, p);

export const IconGrid = (p) => wrap(<>
  <path d="M3 3h18v18H3z" /><path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
</>, p);

export const IconRoom = (p) => wrap(<>
  <path d="M3 21V9l9-6 9 6v12" /><path d="M3 21h18M9 21v-7h6v7" />
</>, p);

export const IconLink = (p) => wrap(<>
  <path d="M10 13a4 4 0 0 0 5.7.4l2.6-2.6a4 4 0 0 0-5.7-5.7L11.2 6.5" />
  <path d="M14 11a4 4 0 0 0-5.7-.4l-2.6 2.6a4 4 0 0 0 5.7 5.7l1.4-1.4" />
</>, p);

export const IconChevron = (p) => wrap(<><path d="m9 5 7 7-7 7" /></>, p);

export const IconSettings = (p) => wrap(<>
  <circle cx="12" cy="12" r="3" />
  <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1" />
</>, p);

export const IconQr = (p) => wrap(<>
  <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" />
  <rect x="3" y="14" width="7" height="7" rx="1" /><path d="M14 14h3v3h-3zM19 19h2v2h-2zM14 19h2v2h-2zM19 14h2v2h-2z" />
</>, p);

export const IconHeart = (p) => wrap(<>
  <path d="M12 20s-7-4.6-7-9.4A4 4 0 0 1 12 7a4 4 0 0 1 7 3.6C19 15.4 12 20 12 20z" />
</>, p);

export const IconHeartFilled = (p) => (
  <svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor" {...p}>
    <path d="M12 20.4s-7.4-4.9-7.4-10A4.4 4.4 0 0 1 12 6.6a4.4 4.4 0 0 1 7.4 3.8c0 5.1-7.4 10-7.4 10z" />
  </svg>
);

export const IconFullscreen = (p) => wrap(<>
  <path d="M4 9V5a1 1 0 0 1 1-1h4M20 9V5a1 1 0 0 0-1-1h-4M4 15v4a1 1 0 0 0 1 1h4M20 15v4a1 1 0 0 1-1 1h-4" />
</>, p);

export const IconExitFullscreen = (p) => wrap(<>
  <path d="M9 4v4a1 1 0 0 1-1 1H4M15 4v4a1 1 0 0 0 1 1h4M9 20v-4a1 1 0 0 0-1-1H4M15 20v-4a1 1 0 0 1 1-1h4" />
</>, p);

export const IconMail = (p) => wrap(<>
  <rect x="3" y="5" width="18" height="14" rx="2" /><path d="m3.5 7 8.5 6 8.5-6" />
</>, p);

export const IconCalculator = (p) => wrap(<>
  <rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 7h8" />
  <path d="M8.5 12h.01M12 12h.01M15.5 12h.01M8.5 16h.01M12 16h.01M15.5 16h.01" />
</>, p);

export const IconBookmark = (p) => wrap(<>
  <path d="M6 4h12v17l-6-4.5L6 21z" />
</>, p);

export const IconSort = (p) => wrap(<>
  <path d="M7 4v16M7 20l-3-3M7 20l3-3M17 20V4M17 4l-3 3M17 4l3 3" />
</>, p);

export const IconStore = (p) => wrap(<>
  <path d="M4 10v9a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-9" />
  <path d="M3 6.5 4.5 3h15L21 6.5a3 3 0 0 1-6 0 3 3 0 0 1-6 0 3 3 0 0 1-6 0z" />
</>, p);

export const IconFile = (p) => wrap(<>
  <path d="M13 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V8z" /><path d="M13 3v5h5" />
</>, p);

export const IconPin = (p) => wrap(<>
  <path d="M12 21s7-6.1 7-11a7 7 0 1 0-14 0c0 4.9 7 11 7 11z" /><circle cx="12" cy="10" r="2.5" />
</>, p);

export const IconGlobe = (p) => wrap(<>
  <circle cx="12" cy="12" r="9" /><path d="M3 12h18" />
  <path d="M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18z" />
</>, p);

export const IconChart = (p) => wrap(<>
  <path d="M4 20V4" /><path d="M4 20h16" />
  <rect x="7.5" y="12" width="3" height="5" rx=".6" />
  <rect x="12.5" y="8" width="3" height="9" rx=".6" />
  <rect x="17" y="14" width="3" height="3" rx=".6" />
</>, p);

export const IconLock = (p) => wrap(<>
  <rect x="4" y="10" width="16" height="10" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" />
</>, p);

export const IconCube = (p) => wrap(<>
  <path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z" /><path d="m4 7.5 8 4.5 8-4.5M12 12v9" />
</>, p);
