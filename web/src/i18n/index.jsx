import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react';
import { LANGUAGES, LANGUAGE_BY_CODE, preferredLanguage } from './languages.js';
import en from './locales/en.js';

/**
 * Translation.
 *
 * English is bundled because it is the fallback and something has to render on
 * the first paint; every other language is a dynamic import, so a French
 * visitor downloads French and nothing else. A key with no translation falls
 * back to English rather than showing the key -- a missing string should look
 * like a missed translation, never like a broken page.
 *
 * `t('a.b', { n: 3 })` interpolates `{n}`.
 */
const I18nCtx = createContext({ t: (k) => k, lang: 'en', setLang: () => {}, dir: 'ltr' });

const STORAGE_KEY = 'visualizer-lang';

const loaders = {
  ar: () => import('./locales/ar.js'),
  bg: () => import('./locales/bg.js'),
  ca: () => import('./locales/ca.js'),
  cs: () => import('./locales/cs.js'),
  da: () => import('./locales/da.js'),
  de: () => import('./locales/de.js'),
  es: () => import('./locales/es.js'),
  et: () => import('./locales/et.js'),
  fi: () => import('./locales/fi.js'),
  fil: () => import('./locales/fil.js'),
  fr: () => import('./locales/fr.js'),
  hr: () => import('./locales/hr.js'),
  hu: () => import('./locales/hu.js'),
  id: () => import('./locales/id.js'),
  it: () => import('./locales/it.js'),
  ja: () => import('./locales/ja.js'),
  nl: () => import('./locales/nl.js'),
  pl: () => import('./locales/pl.js'),
  pt: () => import('./locales/pt.js'),
  ro: () => import('./locales/ro.js'),
  ru: () => import('./locales/ru.js'),
  sv: () => import('./locales/sv.js'),
  tr: () => import('./locales/tr.js'),
};

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(() => {
    try { return preferredLanguage(localStorage.getItem(STORAGE_KEY)); } catch { return 'en'; }
  });
  const [dict, setDict] = useState(en);

  useEffect(() => {
    let alive = true;
    if (lang === 'en' || !loaders[lang]) { setDict(en); return undefined; }
    loaders[lang]()
      .then((m) => { if (alive) setDict(m.default); })
      // A language that fails to load is an English page, not a blank one.
      .catch(() => { if (alive) setDict(en); });
    return () => { alive = false; };
  }, [lang]);

  const dir = LANGUAGE_BY_CODE[lang]?.dir ?? 'ltr';

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = dir;
  }, [lang, dir]);

  const setLang = useCallback((next) => {
    setLangState(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* private mode */ }
  }, []);

  const t = useCallback((key, vars) => {
    const raw = dict[key] ?? en[key] ?? key;
    if (!vars) return raw;
    return raw.replace(/\{(\w+)\}/g, (m, k) => (vars[k] ?? m));
  }, [dict]);

  const value = useMemo(() => ({ t, lang, setLang, dir }), [t, lang, setLang, dir]);
  return <I18nCtx.Provider value={value}>{children}</I18nCtx.Provider>;
}

export const useI18n = () => useContext(I18nCtx);
export const useT = () => useContext(I18nCtx).t;

/** The language switcher in the header. */
export default function LanguagePicker() {
  const { lang, setLang } = useI18n();
  return (
    <select
      className="select lang-select"
      value={lang}
      onChange={(e) => setLang(e.target.value)}
      aria-label="Language"
    >
      {LANGUAGES.map((l) => (
        <option key={l.code} value={l.code}>{l.native}</option>
      ))}
    </select>
  );
}
