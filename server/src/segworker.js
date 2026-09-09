/**
 * Detection worker.
 *
 * Owns the segmentation model and does the whole job, replying with just the
 * finished objectList. Lives in its own process so its memory -- and any
 * out-of-memory death -- cannot touch the API.
 */
import { autoDetectSurfaces } from './autodetect.js';

process.on('message', async (msg) => {
  const { id, imagePath, opts } = msg ?? {};
  try {
    const result = await autoDetectSurfaces(imagePath, opts ?? {});
    process.send({ id, ok: true, result });
  } catch (e) {
    process.send({ id, ok: false, error: e?.message ?? String(e) });
  }
});

// Nothing to do until the parent asks; do not hold the event loop open on our
// own account, so killing the parent takes this down too.
process.on('disconnect', () => process.exit(0));
