import React, { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Download, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { Switch } from './ui/switch';
import { Button } from './ui/button';
import { useConfig } from '@/contexts/ConfigContext';

// Mirrors SpeakerModelStatus in src-tauri/src/audio/speaker_tracker.rs
interface SpeakerModelStatus {
  name: string;
  file: string;
  downloaded: boolean;
  size_bytes: number | null;
}

export function SpeakerSettings() {
  const { isAutoRetranscribe, toggleIsAutoRetranscribe, betaFeatures } = useConfig();
  const [status, setStatus] = useState<SpeakerModelStatus | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    invoke<SpeakerModelStatus>('get_speaker_model_status')
      .then(setStatus)
      .catch((err) => console.error('Failed to read speaker model status:', err));
  }, []);

  const handleDownload = useCallback(async () => {
    setDownloading(true);
    try {
      setStatus(await invoke<SpeakerModelStatus>('download_speaker_model'));
      toast.success('Speaker model downloaded');
    } catch (err) {
      toast.error(`Speaker model download failed: ${err}`);
    } finally {
      setDownloading(false);
    }
  }, []);

  return (
    <div className="flex flex-col gap-4 mt-4">
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Speaker Identification</h3>
        <p className="text-sm text-gray-600 mb-4">
          Voice model used to tell speakers apart in transcripts.
        </p>

        <div className="flex items-center justify-between rounded-md border border-gray-200 p-4">
          <div>
            <div className="font-medium text-gray-900">{status?.name ?? 'Loading...'}</div>
            {status && <div className="text-xs text-gray-500 mt-1">{status.file}</div>}
          </div>
          {status && (status.downloaded ? (
            <span className="text-sm text-green-700">
              Downloaded{status.size_bytes ? ` (${(status.size_bytes / 1024 / 1024).toFixed(1)} MB)` : ''}
            </span>
          ) : (
            <Button size="sm" variant="outline" onClick={handleDownload} disabled={downloading}>
              {downloading ? <Loader2 className="mr-2 animate-spin" size={16} /> : <Download className="mr-2" size={16} />}
              Download
            </Button>
          ))}
        </div>
        {status && !status.downloaded && (
          <p className="text-xs text-gray-500 mt-2">Downloads automatically on first use.</p>
        )}

        <ul className="text-sm text-gray-600 mt-4 list-disc pl-5 space-y-1">
          <li>Live recording: &quot;Me&quot; from your microphone, other voices grouped on the fly.</li>
          <li>
            Enhance and import: the whole recording is analysed at once; the number of speakers is
            detected automatically or set in the Enhance dialog.
          </li>
        </ul>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Enhance After Recording</h3>
            <p className="text-sm text-gray-600">
              When a recording stops, retranscribe it with full speaker identification. Auto summary waits for it.
            </p>
            {!betaFeatures.importAndRetranscribe && (
              <p className="text-xs text-amber-700 mt-1">Requires the &quot;Import Audio &amp; Retranscribe&quot; beta feature.</p>
            )}
          </div>
          <Switch
            checked={isAutoRetranscribe && betaFeatures.importAndRetranscribe}
            onCheckedChange={toggleIsAutoRetranscribe}
            disabled={!betaFeatures.importAndRetranscribe}
          />
        </div>
      </div>
    </div>
  );
}
