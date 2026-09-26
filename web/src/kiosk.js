import { useEffect, useRef, useState } from 'react';

/**
 * Showroom kiosk mode.
 *
 * A screen on a shop floor is used by a stranger every few minutes and nobody
 * tidies up between them. Two things follow from that and nothing else really
 * does: the way out to the admin panel has to be gone, and the thing has to
 * put itself back to the start when the last person walks away -- otherwise by
 * lunchtime it is showing somebody else's half-finished bathroom.
 *
 * Entered by URL (`?kiosk=1`) and remembered for the tab, so the shop can point
 * a browser at one address on boot and leave it. It is a display mode, not a
 * security boundary: the admin panel is protected by its own sign-in.
 */
const KEY = 'kiosk-mode';

export function isKiosk() {
  try {
    const q = new URLSearchParams(window.location.search);
    if (q.get('kiosk') === '1') {
      sessionStorage.setItem(KEY, '1');
      return true;
    }
    if (q.get('kiosk') === '0') {
      sessionStorage.removeItem(KEY);
      return false;
    }
    return sessionStorage.getItem(KEY) === '1';
  } catch {
    return false;
  }
}

/**
 * Reset after a stretch of no input.
 *
 * `onIdle` fires once per idle period, not repeatedly -- the timer only starts
 * again after somebody touches the screen, so an unattended kiosk sits quietly
 * on the room list instead of navigating in a loop all night.
 */
export function useIdleReset(seconds, onIdle, enabled = true) {
  const cb = useRef(onIdle);
  cb.current = onIdle;
  const [idle, setIdle] = useState(false);

  useEffect(() => {
    if (!enabled || !seconds) return undefined;
    let timer = null;
    let fired = false;

    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        fired = true;
        setIdle(true);
        cb.current?.();
      }, seconds * 1000);
    };

    const wake = () => {
      if (fired) { fired = false; setIdle(false); }
      arm();
    };

    const events = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'];
    for (const e of events) window.addEventListener(e, wake, { passive: true });
    arm();

    return () => {
      clearTimeout(timer);
      for (const e of events) window.removeEventListener(e, wake);
    };
  }, [seconds, enabled]);

  return idle;
}
