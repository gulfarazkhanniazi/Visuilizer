/**
 * Runs surface detection in a child process.
 *
 * The segmentation model needs a few hundred megabytes. Held inside the API
 * process, a machine that runs short of memory takes the whole server down
 * with it -- the app dies, not just the feature. In its own process the worst
 * case is a failed detection and a clear message, and the memory can be handed
 * back when nobody is uploading.
 *
 * The child does the whole job (segmentation, geometry, polygon tracing) and
 * returns the finished objectList, so only a few kB ever crosses the IPC
 * boundary rather than full-resolution masks.
 */
import { fork } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WORKER = path.join(__dirname, 'segworker.js');

// Long enough that a burst of uploads reuses the warm model, short enough that
// an idle server is not sitting on hundreds of megabytes.
const IDLE_MS = 5 * 60 * 1000;
const JOB_TIMEOUT_MS = 180 * 1000;
// Generous: the point of the child process is isolation, not a tight budget.
// Too low a cap turns a survivable spike into a self-inflicted crash.
const HEAP_MB = 3072;

let child = null;
let idleTimer = null;
let nextId = 1;
const pending = new Map();
let lastStderr = '';

function stopWorker() {
  clearTimeout(idleTimer);
  idleTimer = null;
  if (child) {
    const c = child;
    child = null;
    c.removeAllListeners();
    c.kill();
  }
}

function scheduleIdleStop() {
  if (pending.size) return;
  clearTimeout(idleTimer);
  idleTimer = setTimeout(stopWorker, IDLE_MS);
}

/**
 * Say what actually happened.
 *
 * Reporting every crash as "out of memory" is a guess dressed up as a
 * diagnosis, and it sends people to close applications when the real fault is
 * something else entirely. Only claim memory when the evidence says so.
 */
function describeExit(code, signal, stderr) {
  const text = (stderr || '').toLowerCase();
  const oom = signal === 'SIGKILL'
    || signal === 'SIGABRT'
    || text.includes('heap out of memory')
    || text.includes('allocation failed')
    || text.includes('bad_alloc');

  if (oom) {
    return 'The detector ran out of memory. Close some applications and try again, '
      + 'or mark the surfaces by hand in the Studio.';
  }

  // 0xC0000142 STATUS_DLL_INIT_FAILED. The child dies before running a line of
  // JavaScript, so there is never any stderr to go on. In practice this means
  // this server process can no longer spawn children at all -- usually because
  // it was orphaned when its parent was killed -- and no amount of retrying
  // will help. Restarting the server is the fix, so say so.
  if (code === 3221225794 || code === -1073741502) {
    return 'The server can no longer start helper processes, so detection cannot run. '
      + 'Restart the server (stop it, then "npm start") and try again. '
      + 'In the meantime you can mark the surfaces by hand in the Studio.';
  }
  const detail = (stderr || '').split('\n').filter(Boolean).slice(-2).join(' ').trim();
  const how = signal ? `signal ${signal}` : `exit code ${code}`;
  return detail
    ? `The detector stopped (${how}): ${detail.slice(0, 300)}`
    : `The detector stopped unexpectedly (${how}).`;
}

function getWorker() {
  if (child) return child;
  lastStderr = '';

  child = fork(WORKER, [], {
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    execArgv: [`--max-old-space-size=${HEAP_MB}`],
  });
  const self = child;

  self.on('message', (msg) => {
    const job = pending.get(msg.id);
    if (!job) return;
    pending.delete(msg.id);
    clearTimeout(job.timer);
    if (msg.ok) job.resolve(msg.result);
    else job.reject(new Error(msg.error || 'Detection failed'));
    scheduleIdleStop();
  });

  self.on('exit', (code, signal) => {
    if (child === self) child = null;
    const message = describeExit(code, signal, lastStderr);
    for (const [, job] of pending) {
      clearTimeout(job.timer);
      job.onWorkerDeath(message);
    }
    pending.clear();
  });

  self.on('error', (err) => {
    if (child === self) child = null;
    for (const [, job] of pending) {
      clearTimeout(job.timer);
      job.onWorkerDeath(`The detector could not be started: ${err.message}`);
    }
    pending.clear();
  });

  // Keep the tail of anything the worker printed; it is the only evidence
  // available once the process is gone.
  self.stderr?.on('data', (b) => {
    const text = String(b);
    lastStderr = (lastStderr + text).slice(-4000);
    const line = text.trim();
    if (line) console.error('[detect]', line.slice(0, 400));
  });

  return self;
}

function runOnce(imagePath, opts, onWorkerDeath) {
  const worker = getWorker();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      stopWorker();               // a wedged worker is not recoverable
      reject(new Error('Detection timed out.'));
    }, JOB_TIMEOUT_MS);

    pending.set(id, { resolve, reject, timer, onWorkerDeath: (m) => onWorkerDeath(m, reject) });
    clearTimeout(idleTimer);
    worker.send({ id, imagePath, opts });
  });
}

/**
 * Detect floor and walls in a photograph.
 *
 * Retries once on a worker death: a fork that loses a race with a momentary
 * memory spike usually succeeds on a fresh process, and asking someone to
 * re-upload their photo for that is a poor trade.
 */
export async function detectSurfaces(imagePath, opts = {}) {
  for (let attempt = 0; attempt < 2; attempt++) {
    const last = attempt === 1;
    try {
      return await runOnce(imagePath, opts, (message, reject) => {
        reject(Object.assign(new Error(message), { workerDied: true }));
      });
    } catch (e) {
      if (!e.workerDied || last) throw e;
      stopWorker();
      // Give the OS a moment to reclaim before trying again.
      await new Promise((r) => setTimeout(r, 1200));
    }
  }
  throw new Error('Detection failed.');
}

export function shutdownDetector() {
  stopWorker();
}
