import React from 'react';
import { Users } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

// Mirrors MAX_SPEAKERS in src-tauri/src/audio/diarization.rs (checked by a Rust test)
export const MAX_SPEAKERS = 8;

/** 'auto' or a number of speakers as text; see `toNumSpeakers` for the command argument. */
export type SpeakerCount = string;

export function toNumSpeakers(value: SpeakerCount): number | null {
  return value === 'auto' ? null : Number(value);
}

export function SpeakerCountSelect({ value, onChange }: { value: SpeakerCount; onChange: (value: SpeakerCount) => void }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Users className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Speakers</span>
      </div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="auto">Detect automatically</SelectItem>
          {Array.from({ length: MAX_SPEAKERS }, (_, i) => i + 1).map((n) => (
            <SelectItem key={n} value={String(n)}>{n}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        Setting the exact number of speakers improves speaker separation
      </p>
    </div>
  );
}
