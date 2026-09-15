// Action palette: validated colorblind-safe on the dark surface (#161C25).
// Mirrors ACTION_COLORS in server.py so the burned-in overlay and the UI agree.
export const ACTIONS = ['STANDING', 'SITTING', 'LYING DOWN', 'FIGHTING', 'UNKNOWN']
export const ACTION_COLOR = {
  'STANDING': '#22A68A',
  'SITTING': '#C08A14',
  'LYING DOWN': '#D94366',
  'FIGHTING': '#8371EA',
  'UNKNOWN': '#6B7684',
}
export const ACTION_GLYPH = {
  'STANDING': '│',
  'SITTING': '┘',
  'LYING DOWN': '—',
  'FIGHTING': '✕',
  'UNKNOWN': '?',
}
export const ALERT_ACTIONS = new Set(['LYING DOWN', 'FIGHTING'])
export const STREAM_SLOTS = ['stream_0', 'stream_1', 'stream_2', 'stream_3']
