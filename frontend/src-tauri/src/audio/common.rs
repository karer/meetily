use crate::api::TranscriptSegment;
use anyhow::Result;
use log::{debug, info};
use once_cell::sync::Lazy;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
use tauri::{AppHandle, Manager, Runtime};
use uuid::Uuid;

use super::diarization;
use super::speaker_tracker;
use super::vad::SpeechSegment;

static ENGINE_LIFECYCLE_LOCK: Lazy<Arc<AsyncMutex<()>>> =
    Lazy::new(|| Arc::new(AsyncMutex::new(())));

pub(crate) async fn acquire_engine_lifecycle_lock() -> OwnedMutexGuard<()> {
    ENGINE_LIFECYCLE_LOCK.clone().lock_owned().await
}

/// Unload the transcription engine after a batch job (import or retranscription).
/// Skips unloading if a live recording is currently in progress, since recording
/// uses the same global engine instances.
pub(crate) async fn unload_engine_after_batch(use_parakeet: bool) {
    let _engine_lifecycle_guard = acquire_engine_lifecycle_lock().await;

    if crate::audio::recording_commands::is_recording().await {
        log::info!("Skipping model unload after batch: recording in progress");
        return;
    }

    if use_parakeet {
        use crate::parakeet_engine::commands::PARAKEET_ENGINE;
        let engine = {
            let guard = PARAKEET_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().cloned()
        };
        if let Some(e) = engine {
            e.unload_model().await;
        }
    } else {
        use crate::whisper_engine::commands::WHISPER_ENGINE;
        let engine = {
            let guard = WHISPER_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().cloned()
        };
        if let Some(e) = engine {
            e.unload_model().await;
        }
    }
}

/// Segments longer than this are split at silence for better transcription quality.
const MAX_SEGMENT_SAMPLES: usize = 25 * 16000; // 25 seconds at 16kHz

/// Prepares a batch job's VAD segments for transcription: diarizes the whole
/// recording into single-speaker pieces ("Speaker N"), then splits long pieces
/// at silence. Without the embedding model the segments keep no speaker.
/// `on_progress(done, total)` reports diarization progress; false cancels.
pub(crate) async fn batch_speaker_pieces<R: Runtime>(
    app: &AppHandle<R>,
    segments: Vec<SpeechSegment>,
    num_speakers: Option<usize>,
    on_progress: impl FnMut(usize, usize) -> bool + Send + 'static,
) -> Result<Vec<(SpeechSegment, Option<String>)>> {
    let model = match app.path().app_data_dir() {
        Ok(dir) => {
            let path = speaker_tracker::model_path(&dir);
            match speaker_tracker::ensure_model(&path).await {
                Ok(()) => Some(path),
                Err(e) => {
                    log::warn!("Speaker embedding model unavailable, no speaker labels: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            log::warn!("No app data dir, no speaker labels: {}", e);
            None
        }
    };

    let labelled: Vec<(SpeechSegment, Option<String>)> = match model {
        Some(path) => tokio::task::spawn_blocking(move || -> Result<_> {
            let mut extractor = match speaker_tracker::new_extractor(&path, 4) {
                Ok(extractor) => extractor,
                Err(e) => {
                    log::warn!("Speaker embedding extractor unavailable, no speaker labels: {}", e);
                    return Ok(segments.into_iter().map(|s| (s, None)).collect());
                }
            };
            let pieces = diarization::diarize(&mut extractor, &segments, num_speakers, on_progress)?;
            let speakers = pieces.iter().map(|p| p.speaker + 1).max().unwrap_or(0);
            info!("Diarization: {} segments -> {} pieces, {} speakers", segments.len(), pieces.len(), speakers);
            Ok(pieces
                .into_iter()
                .map(|p| (p.segment, Some(format!("Speaker {}", p.speaker + 1))))
                .collect())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Diarization task panicked: {}", e))??,
        None => segments.into_iter().map(|s| (s, None)).collect(),
    };

    Ok(labelled
        .into_iter()
        .flat_map(|(segment, speaker)| {
            split_segment_at_silence(&segment, MAX_SEGMENT_SAMPLES)
                .into_iter()
                .map(move |s| (s, speaker.clone()))
        })
        .collect())
}

/// Gives the "Me" label of the replaced transcript (decided live from the
/// microphone) to the diarized speaker that mostly overlaps it in time.
/// `old` is (start_s, end_s, speaker); `new` is (text, start_ms, end_ms, speaker).
pub(crate) fn carry_over_me(old: &[(f64, f64, String)], new: &mut [(String, f64, f64, Option<String>)]) {
    let me: Vec<(f64, f64)> = old
        .iter()
        .filter(|(_, _, s)| s == speaker_tracker::SPEAKER_ME)
        .map(|&(a, b, _)| (a * 1000.0, b * 1000.0))
        .collect();
    if me.is_empty() {
        return;
    }
    // speaker -> (total ms, ms overlapping old "Me")
    let mut stats: std::collections::HashMap<String, (f64, f64)> = Default::default();
    for (_, start, end, speaker) in new.iter() {
        let Some(speaker) = speaker else { continue };
        let overlap: f64 = me.iter().map(|&(a, b)| (end.min(b) - start.max(a)).max(0.0)).sum();
        let entry = stats.entry(speaker.clone()).or_default();
        entry.0 += end - start;
        entry.1 += overlap;
    }
    let best = stats
        .iter()
        .filter(|(_, (total, overlap))| *overlap > 0.5 * total)
        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        .map(|(speaker, _)| speaker.clone());
    if let Some(best) = best {
        info!("Diarized {} matches the live \"Me\" label", best);
        for (_, _, _, speaker) in new.iter_mut() {
            if speaker.as_deref() == Some(best.as_str()) {
                *speaker = Some(speaker_tracker::SPEAKER_ME.to_string());
            }
        }
    }
}

/// Create transcript segments from transcription results.
/// Each tuple is (text, start_ms, end_ms, speaker) from VAD timestamps.
pub(crate) fn create_transcript_segments(transcripts: &[(String, f64, f64, Option<String>)]) -> Vec<TranscriptSegment> {
    transcripts
        .iter()
        .map(|(text, start_ms, end_ms, speaker)| {
            let start_seconds = start_ms / 1000.0;
            let end_seconds = end_ms / 1000.0;
            let duration = end_seconds - start_seconds;

            TranscriptSegment {
                id: format!("transcript-{}", Uuid::new_v4()),
                text: text.trim().to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                audio_start_time: Some(start_seconds),
                audio_end_time: Some(end_seconds),
                duration: Some(duration),
                speaker: speaker.clone(),
            }
        })
        .collect()
}

/// Write transcripts.json to a meeting folder (atomic write with temp file)
pub(crate) fn write_transcripts_json(folder: &Path, segments: &[TranscriptSegment]) -> Result<()> {
    let transcript_path = folder.join("transcripts.json");
    let temp_path = folder.join(".transcripts.json.tmp");

    let json = serde_json::json!({
        "version": "1.0",
        "last_updated": chrono::Utc::now().to_rfc3339(),
        "total_segments": segments.len(),
        "segments": segments.iter().enumerate().map(|(i, s)| {
            serde_json::json!({
                "id": s.id,
                "text": s.text,
                "timestamp": s.timestamp,
                "audio_start_time": s.audio_start_time,
                "audio_end_time": s.audio_end_time,
                "duration": s.duration,
                "speaker": s.speaker,
                "sequence_id": i
            })
        }).collect::<Vec<_>>()
    });

    let json_string = serde_json::to_string_pretty(&json)?;
    std::fs::write(&temp_path, &json_string)?;
    std::fs::rename(&temp_path, &transcript_path)?;

    info!(
        "Wrote transcripts.json with {} segments to {}",
        segments.len(),
        transcript_path.display()
    );
    Ok(())
}

/// Split a long speech segment at the lowest-energy (silence) point near the target size.
///
/// Scans for 100ms windows with minimal RMS energy within +/-3 seconds of each target
/// split point. If no clear silence is found, falls back to a 1-second overlap split
/// to avoid cutting words at boundaries.
pub(crate) fn split_segment_at_silence(
    segment: &crate::audio::vad::SpeechSegment,
    max_samples: usize,
) -> Vec<crate::audio::vad::SpeechSegment> {
    const SAMPLE_RATE: usize = 16000;
    // 100ms window for energy measurement (1600 samples at 16kHz)
    const ENERGY_WINDOW: usize = SAMPLE_RATE / 10;
    // Search +/-3 seconds around the target split point
    const SEARCH_RADIUS: usize = SAMPLE_RATE * 3;
    // RMS threshold below which we consider a window "silent"
    const SILENCE_RMS_THRESHOLD: f32 = 0.02;
    // Overlap to use when no silence boundary is found (1 second)
    const FALLBACK_OVERLAP: usize = SAMPLE_RATE;

    let total = segment.samples.len();
    if total <= max_samples {
        return vec![segment.clone()];
    }

    let ms_per_sample = (segment.end_timestamp_ms - segment.start_timestamp_ms)
        / segment.samples.len() as f64;
    let mut result = Vec::new();
    let mut pos = 0usize;

    while pos < total {
        let remaining = total - pos;
        if remaining <= max_samples {
            // Last chunk - take everything remaining
            let chunk_samples = segment.samples[pos..].to_vec();
            let chunk_start_ms = segment.start_timestamp_ms + (pos as f64 * ms_per_sample);
            let chunk_end_ms = segment.end_timestamp_ms;
            result.push(crate::audio::vad::SpeechSegment {
                samples: chunk_samples,
                start_timestamp_ms: chunk_start_ms,
                end_timestamp_ms: chunk_end_ms,
                confidence: segment.confidence,
            });
            break;
        }

        // Target split point
        let target = pos + max_samples;

        // Search window: [target - SEARCH_RADIUS, target + SEARCH_RADIUS]
        let search_start = target.saturating_sub(SEARCH_RADIUS).max(pos + SAMPLE_RATE);
        let search_end = (target + SEARCH_RADIUS).min(total.saturating_sub(ENERGY_WINDOW));

        // Find the lowest-energy 100ms window in the search range
        let mut best_split = target.min(total); // fallback: exact target
        let mut best_rms = f32::MAX;

        if search_start + ENERGY_WINDOW <= search_end {
            let mut idx = search_start;
            while idx + ENERGY_WINDOW <= search_end {
                let window = &segment.samples[idx..idx + ENERGY_WINDOW];
                let rms = (window.iter().map(|s| s * s).sum::<f32>() / ENERGY_WINDOW as f32).sqrt();
                if rms < best_rms {
                    best_rms = rms;
                    best_split = idx + ENERGY_WINDOW / 2; // split at center of quiet window
                }
                // Step by 10ms (160 samples) for efficiency
                idx += SAMPLE_RATE / 100;
            }
        }

        let split_at = best_split;
        if best_rms <= SILENCE_RMS_THRESHOLD {
            debug!(
                "Splitting at silence boundary: sample {} (RMS={:.4})",
                split_at, best_rms
            );
        } else {
            debug!(
                "No silence found near target (best RMS={:.4}), splitting with overlap at sample {}",
                best_rms, split_at
            );
        }

        // Determine the actual end of this chunk (with overlap if no silence)
        let chunk_end = if best_rms > SILENCE_RMS_THRESHOLD {
            (split_at + FALLBACK_OVERLAP).min(total)
        } else {
            split_at
        };

        let chunk_samples = segment.samples[pos..chunk_end].to_vec();
        let chunk_start_ms = segment.start_timestamp_ms + (pos as f64 * ms_per_sample);
        let chunk_end_ms = segment.start_timestamp_ms + (chunk_end as f64 * ms_per_sample);

        result.push(crate::audio::vad::SpeechSegment {
            samples: chunk_samples,
            start_timestamp_ms: chunk_start_ms,
            end_timestamp_ms: chunk_end_ms,
            confidence: segment.confidence,
        });

        // Advance position to where the current chunk actually ends
        // to avoid transcribing the overlap region twice
        pos = chunk_end;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(start_ms: f64, end_ms: f64, speaker: &str) -> (String, f64, f64, Option<String>) {
        (String::new(), start_ms, end_ms, Some(speaker.to_string()))
    }

    #[test]
    fn me_moves_to_the_overlapping_speaker() {
        let old = vec![(0.0, 10.0, "Me".to_string()), (10.0, 20.0, "Speaker 1".to_string())];
        let mut new = vec![row(0.0, 9000.0, "Speaker 2"), row(9000.0, 20000.0, "Speaker 1"), row(20000.0, 25000.0, "Speaker 2")];
        carry_over_me(&old, &mut new);
        let speakers: Vec<_> = new.iter().map(|r| r.3.clone().unwrap()).collect();
        assert_eq!(speakers, vec!["Me", "Speaker 1", "Me"]);
    }

    #[test]
    fn me_needs_majority_overlap() {
        let old = vec![(0.0, 2.0, "Me".to_string())];
        let mut new = vec![row(0.0, 10000.0, "Speaker 1")];
        carry_over_me(&old, &mut new);
        assert_eq!(new[0].3.as_deref(), Some("Speaker 1"));
    }

    #[tokio::test]
    async fn test_engine_lifecycle_lock_serializes_acquirers() {
        let guard = acquire_engine_lifecycle_lock().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (acquired_tx, mut acquired_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async {
            started_tx.send(()).unwrap();
            let _guard = acquire_engine_lifecycle_lock().await;
            acquired_tx.send(()).unwrap();
        });

        started_rx.await.unwrap();
        assert!(acquired_rx.try_recv().is_err());
        drop(guard);

        acquired_rx.await.unwrap();
        waiter.await.unwrap();
    }
}
