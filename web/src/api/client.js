const BASE = '/api';
const TOKEN_KEY = 'admin-token';

/** The admin session token, if this browser is signed in. */
export const authToken = () => localStorage.getItem(TOKEN_KEY);

export function setAuthToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

/**
 * Fired when the server rejects a token -- expired, revoked, or from a
 * database that has since been replaced. The admin screen listens for it and
 * shows the sign-in form rather than a wall of failed requests.
 */
export const AUTH_LOST = 'visualizer:auth-lost';

async function request(path, options = {}) {
  // Every request carries the visitor id, so saved schemes and the wishlist
  // follow the browser without anyone having to create an account. The admin
  // token is separate and only present once someone has signed in.
  const headers = { ...(options.headers ?? {}), 'x-visitor': visitorId() };
  const token = authToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(BASE + path, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    if (res.status === 401 && body.needsAuth) {
      setAuthToken(null);
      window.dispatchEvent(new CustomEvent(AUTH_LOST));
    }
    throw new Error(body.error || `${res.status} ${res.statusText}`);
  }
  return res.status === 204 ? null : res.json();
}

const json = (method, body) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  vendor: () => request('/vendor'),
  saveVendor: (v) => request('/vendor', json('PUT', v)),
  uploadLogo: (file) => {
    const fd = new FormData();
    fd.append('logo', file);
    return request('/vendor/logo', { method: 'POST', body: fd });
  },

  roomCategories: () => request('/room-categories'),
  saveRoomCategory: (c) => request('/room-categories', json('POST', c)),
  rooms: (params = {}) => request(`/rooms?${new URLSearchParams(params)}`),
  room: (id) => request(`/rooms/${id}`),
  saveRoom: (id, data) => request(`/rooms/${id}`, json('PUT', data)),
  autoDetect: (id, opts = {}) => request(`/rooms/${id}/auto-detect`, json('POST', opts)),
  deleteRoom: (id) => request(`/rooms/${id}`, { method: 'DELETE' }),
  duplicateRoom: (id) => request(`/rooms/${id}/duplicate`, { method: 'POST' }),
  uploadModel: (file, fields = {}) => {
    const fd = new FormData();
    fd.append('model', file);
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return request('/rooms/upload-model', { method: 'POST', body: fd });
  },
  uploadRoom: (file, fields = {}) => {
    const fd = new FormData();
    fd.append('photo', file);
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return request('/rooms/upload', { method: 'POST', body: fd });
  },

  productCategories: () => request('/product-categories'),
  saveProductCategory: (c) => request('/product-categories', json('POST', c)),
  products: (params = {}) => request(`/products?${new URLSearchParams(params)}`),
  product: (id) => request(`/products/${id}`),
  createProduct: (fields, files = []) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) {
      fd.append(k, typeof v === 'object' ? JSON.stringify(v) : v);
    }
    for (const f of files) fd.append('faces', f);
    return request('/products', { method: 'POST', body: fd });
  },
  updateProduct: (id, fields, files = []) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) {
      fd.append(k, typeof v === 'object' ? JSON.stringify(v) : v);
    }
    for (const f of files) fd.append('faces', f);
    return request(`/products/${id}`, { method: 'PUT', body: fd });
  },
  deleteProduct: (id) => request(`/products/${id}`, { method: 'DELETE' }),
  bulkProducts: (files, fields = {}) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) {
      fd.append(k, typeof v === 'object' ? JSON.stringify(v) : v);
    }
    for (const f of files) fd.append('images', f);
    return request('/products/bulk', { method: 'POST', body: fd });
  },

  savedRooms: () => request('/saved-rooms'),
  // Named for what it saves. It used to be `saveRoom`, which silently replaced
  // the room-editing `saveRoom` above it in this same object literal -- so the
  // Studio's Save button was posting its surfaces to the visitor scheme
  // endpoint and getting a 400 back.
  saveScheme: (roomId, name, payload, preview) =>
    request('/saved-rooms', json('POST', { roomId, name, payload, preview })),
  deleteSavedRoom: (id) => request(`/saved-rooms/${id}`, { method: 'DELETE' }),

  wishlist: () => request('/wishlist'),
  addWishlist: (id) => request(`/wishlist/${id}`, { method: 'POST' }),
  removeWishlist: (id) => request(`/wishlist/${id}`, { method: 'DELETE' }),

  stores: () => request('/stores'),
  saveStore: (s) => request('/stores', json('POST', s)),
  deleteStore: (id) => request(`/stores/${id}`, { method: 'DELETE' }),

  share: (payload, preview) => request('/share', json('POST', { payload, preview })),
  getShare: (code) => request(`/share/${code}`),
  createLead: (lead) => request('/leads', json('POST', lead)),
  leads: () => request('/leads'),

  authStatus: () => request('/auth/status'),
  login: (email, password) => request('/auth/login', json('POST', { email, password })),
  logout: () => request('/auth/logout', { method: 'POST' }),
  bootstrap: (fields) => request('/auth/bootstrap', json('POST', fields)),
  users: () => request('/auth/users'),
  createUser: (fields) => request('/auth/users', json('POST', fields)),
  deleteUser: (id) => request(`/auth/users/${id}`, { method: 'DELETE' }),
  changePassword: (currentPassword, newPassword) =>
    request('/auth/password', json('POST', { currentPassword, newPassword })),

  analytics: (days = 30) => request(`/analytics?days=${days}`),
  pruneAnalytics: (keepDays) => request('/analytics/prune', json('POST', { keepDays })),
};

/* ------------------------------------------------------------ analytics -- */

/**
 * Event recording.
 *
 * Batched because a visualizer session generates a lot of small events -- every
 * product applied is one -- and a request each would be more traffic than the
 * renders. Flushed on a short timer, when the batch fills, and on the way out
 * with sendBeacon, which is the only thing that survives a page being closed.
 */
const queue = [];
let flushTimer = null;

function sessionKey() {
  let id = sessionStorage.getItem('viz-session');
  if (!id) {
    id = `s_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
    sessionStorage.setItem('viz-session', id);
  }
  return id;
}

function flush(beacon = false) {
  if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
  if (!queue.length) return;
  const events = queue.splice(0, queue.length);
  const body = JSON.stringify({ events, visitor: visitorId() });

  if (beacon && navigator.sendBeacon) {
    navigator.sendBeacon(`${BASE}/events`, new Blob([body], { type: 'application/json' }));
    return;
  }
  fetch(`${BASE}/events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-visitor': visitorId() },
    body,
    keepalive: true,
  }).catch(() => {});
}

export function track(type, fields = {}) {
  queue.push({ type, session: sessionKey(), ...fields });
  if (queue.length >= 12) return flush();
  if (!flushTimer) flushTimer = setTimeout(flush, 4000);
  return undefined;
}

if (typeof window !== 'undefined') {
  const leave = () => flush(true);
  window.addEventListener('pagehide', leave);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') leave();
  });
}

/** A stable per-browser id, so a visitor's uploaded rooms follow them back. */
export function visitorId() {
  let id = localStorage.getItem('visitor-id');
  if (!id) {
    id = `v_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
    localStorage.setItem('visitor-id', id);
  }
  return id;
}
