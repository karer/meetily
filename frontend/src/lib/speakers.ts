// Speaker labels arrive from the Rust core as display text ("Me", "Speaker 1",
// or a name the user gave), see src-tauri/src/audio/speaker_tracker.rs.

/** Prefixes a transcript line with its speaker, for copy, export and summaries. */
export function withSpeaker(speaker: string | null | undefined, text: string): string {
  return speaker ? `${speaker}: ${text}` : text;
}

const SPEAKER_COLORS = [
  'text-blue-700 bg-blue-50',
  'text-emerald-700 bg-emerald-50',
  'text-amber-700 bg-amber-50',
  'text-purple-700 bg-purple-50',
  'text-rose-700 bg-rose-50',
  'text-cyan-700 bg-cyan-50',
];

/** Stable color per speaker label, so one person keeps one color across the transcript. */
export function speakerColor(speaker: string): string {
  let hash = 0;
  for (let i = 0; i < speaker.length; i++) {
    hash = (hash * 31 + speaker.charCodeAt(i)) | 0;
  }
  return SPEAKER_COLORS[Math.abs(hash) % SPEAKER_COLORS.length];
}
