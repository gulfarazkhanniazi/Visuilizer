import { configureStore } from '@reduxjs/toolkit';
import viz from './vizSlice.js';
import catalog from './catalogSlice.js';

export const store = configureStore({
  reducer: { viz, catalog },
  middleware: (getDefault) =>
    // Room objects carry plain JSON only, but they are large; skipping the
    // deep serialisability scan keeps surface edits from stuttering.
    getDefault({ serializableCheck: false, immutableCheck: false }),
});
