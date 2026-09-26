/**
 * The languages the visitor-facing app ships in.
 *
 * `dir: 'rtl'` is not decoration -- Arabic reverses the whole layout, and the
 * panel/rail/stage arrangement has to flip with it.
 *
 * Only the visitor-facing surfaces are translated: the room picker, the
 * visualizer and its dialogs. The Studio and the Admin panel stay in English,
 * because they are operated by the business rather than by its customers and
 * translating them would be maintaining a second, larger vocabulary for an
 * audience of one.
 */
export const LANGUAGES = [
  { code: 'en',  name: 'English',    native: 'English' },
  { code: 'ar',  name: 'Arabic',     native: 'العربية', dir: 'rtl' },
  { code: 'bg',  name: 'Bulgarian',  native: 'Български' },
  { code: 'ca',  name: 'Catalan',    native: 'Català' },
  { code: 'cs',  name: 'Czech',      native: 'Čeština' },
  { code: 'da',  name: 'Danish',     native: 'Dansk' },
  { code: 'de',  name: 'German',     native: 'Deutsch' },
  { code: 'es',  name: 'Spanish',    native: 'Español' },
  { code: 'et',  name: 'Estonian',   native: 'Eesti' },
  { code: 'fi',  name: 'Finnish',    native: 'Suomi' },
  { code: 'fil', name: 'Filipino',   native: 'Filipino' },
  { code: 'fr',  name: 'French',     native: 'Français' },
  { code: 'hr',  name: 'Croatian',   native: 'Hrvatski' },
  { code: 'hu',  name: 'Hungarian',  native: 'Magyar' },
  { code: 'id',  name: 'Indonesian', native: 'Bahasa Indonesia' },
  { code: 'it',  name: 'Italian',    native: 'Italiano' },
  { code: 'ja',  name: 'Japanese',   native: '日本語' },
  { code: 'nl',  name: 'Dutch',      native: 'Nederlands' },
  { code: 'pl',  name: 'Polish',     native: 'Polski' },
  { code: 'pt',  name: 'Portuguese', native: 'Português' },
  { code: 'ro',  name: 'Romanian',   native: 'Română' },
  { code: 'ru',  name: 'Russian',    native: 'Русский' },
  { code: 'sv',  name: 'Swedish',    native: 'Svenska' },
  { code: 'tr',  name: 'Turkish',    native: 'Türkçe' },
];

export const LANGUAGE_BY_CODE = Object.fromEntries(LANGUAGES.map((l) => [l.code, l]));

/**
 * Best match for the browser's preference.
 *
 * Matches the base tag, so pt-BR gets Portuguese rather than falling all the
 * way back to English, which is the failure people actually notice.
 */
export function preferredLanguage(stored) {
  if (stored && LANGUAGE_BY_CODE[stored]) return stored;
  for (const tag of navigator.languages ?? [navigator.language ?? 'en']) {
    const base = String(tag).toLowerCase().split('-')[0];
    if (LANGUAGE_BY_CODE[base]) return base;
    if (base === 'in') return 'id';       // legacy code for Indonesian
    if (base === 'tl') return 'fil';      // Tagalog
  }
  return 'en';
}
