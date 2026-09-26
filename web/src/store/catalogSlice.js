import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { api } from '../api/client.js';

export const loadCatalog = createAsyncThunk('catalog/load', async () => {
  const [vendor, productCategories, products, roomCategories, wishlist] = await Promise.all([
    api.vendor(),
    api.productCategories(),
    api.products({ limit: 500 }),
    api.roomCategories(),
    api.wishlist().catch(() => []),
  ]);
  return {
    vendor,
    productCategories,
    products: products.items,
    roomCategories,
    wishlist: wishlist.map((p) => p.id),
  };
});

/** Optimistic wishlist toggle -- the heart must not wait for a round trip. */
export const toggleWishlist = createAsyncThunk(
  'catalog/toggleWishlist',
  async (productId, { getState }) => {
    const on = getState().catalog.wishlist.includes(productId);
    if (on) await api.removeWishlist(productId);
    else await api.addWishlist(productId);
    return { productId, on: !on };
  },
);

const catalogSlice = createSlice({
  name: 'catalog',
  initialState: {
    vendor: null,
    products: [],
    productCategories: [],
    roomCategories: [],
    wishlist: [],
    status: 'idle',
    error: null,
  },
  reducers: {
    upsertProduct(state, { payload }) {
      const i = state.products.findIndex((p) => p.id === payload.id);
      if (i >= 0) state.products[i] = payload;
      else state.products.push(payload);
    },
    removeProduct(state, { payload }) {
      state.products = state.products.filter((p) => p.id !== payload);
    },
    setVendor(state, { payload }) {
      state.vendor = payload;
    },
    setWishlistLocal(state, { payload }) {
      state.wishlist = payload;
    },
  },
  extraReducers: (b) => {
    b.addCase(loadCatalog.pending, (s) => { s.status = 'loading'; });
    b.addCase(loadCatalog.fulfilled, (s, { payload }) => {
      Object.assign(s, payload, { status: 'ready' });
    });
    b.addCase(loadCatalog.rejected, (s, a) => {
      s.status = 'error';
      s.error = a.error.message;
    });
    b.addCase(toggleWishlist.pending, (s, a) => {
      const id = a.meta.arg;
      s.wishlist = s.wishlist.includes(id)
        ? s.wishlist.filter((x) => x !== id)
        : [...s.wishlist, id];
    });
    b.addCase(toggleWishlist.rejected, (s, a) => {
      // Put it back the way it was; the server said no.
      const id = a.meta.arg;
      s.wishlist = s.wishlist.includes(id)
        ? s.wishlist.filter((x) => x !== id)
        : [...s.wishlist, id];
    });
  },
});

export const { upsertProduct, removeProduct, setVendor, setWishlistLocal } = catalogSlice.actions;
export default catalogSlice.reducer;

export const selectProductById = (state, id) =>
  state.catalog.products.find((p) => p.id === id) ?? null;

/** Products a given surface can legitimately take (a floor tile is not a splashback). */
export const selectProductsForSurface = (state, surfaceKey) =>
  state.catalog.products.filter((p) => !surfaceKey || p.surfaces.includes(surfaceKey));
