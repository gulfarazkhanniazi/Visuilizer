import { db } from './db.js';

const rooms = db.prepare('SELECT id, data FROM rooms').all();
const update = db.prepare('UPDATE rooms SET data = ? WHERE id = ?');

let count = 0;
for (const r of rooms) {
  try {
    const data = JSON.parse(r.data);
    let changed = false;
    if (data.objectList) {
      for (const obj of data.objectList) {
        if (obj.defaults) {
          if (obj.defaults.grout && obj.defaults.grout.size !== 0) {
            obj.defaults.grout.size = 0;
            changed = true;
          }
          if (obj.defaults.gloss !== undefined && obj.defaults.gloss !== 0) {
            obj.defaults.gloss = 0;
            changed = true;
          }
          if (obj.defaults.shade !== undefined && obj.defaults.shade !== 0) {
            obj.defaults.shade = 0;
            changed = true;
          }
          if (obj.defaults.detail !== undefined && obj.defaults.detail !== 0) {
            obj.defaults.detail = 0;
            changed = true;
          }
          if (obj.defaults.bevel !== undefined && obj.defaults.bevel !== 0) {
            obj.defaults.bevel = 0;
            changed = true;
          }
        }
      }
    }
    if (changed) {
      update.run(JSON.stringify(data), r.id);
      count++;
    }
  } catch (e) {
    console.error('Error parsing room', r.id, e);
  }
}
console.log(`Updated ${count} existing rooms to have 0 for all requested defaults.`);
