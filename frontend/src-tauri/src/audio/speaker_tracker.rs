//! Speaker labels for transcripts.
//!
//! Mic and system audio are captured as separate streams, so every VAD segment
//! carries both sources (see `SourceAudio`). A segment where the microphone
//! dominates is labelled "Me". Remote speech is split into "Speaker N" by
//! clustering voice embeddings computed with sherpa-onnx and the WeSpeaker
//! ResNet34-LM model (the embedding model used by pyannote 3.x).
//!
//! Without the embedding model, remote speech falls back to "Others".
//! Labels are stored as display text, so renaming a speaker is a plain text update.

use std::path::{Path, PathBuf};
use std::time::Duration;

use log::{info, warn};
use sha2::{Digest, Sha256};
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};

pub const SPEAKER_ME: &str = "Me";
pub const SPEAKER_OTHERS: &str = "Others";

const MODEL_FILE: &str = "wespeaker_en_voxceleb_resnet34_LM.onnx";
const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx";
const MODEL_SHA256: &str = "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012";

pub const SOURCE_SAMPLE_RATE: u32 = 16000;
/// 30ms frames at 16kHz, same granularity as the VAD.
const FRAME_SAMPLES: usize = 480;
/// Frames quieter than this on both sources are ignored.
const SILENCE_RMS: f32 = 0.003;
/// A source must be this many times louder than the other to win a frame.
/// Keeps speaker bleed (remote voice picked up by the mic) from counting as `me`.
const DOMINANCE_RATIO: f32 = 2.0;
/// Below this, embeddings are unreliable and the previous remote speaker is reused.
const MIN_EMBEDDING_SAMPLES: usize = SOURCE_SAMPLE_RATE as usize;
const MAX_EMBEDDING_SAMPLES: usize = SOURCE_SAMPLE_RATE as usize * 20;
/// Cosine similarity needed to join an existing speaker. Measured on distinct
/// voices: same voice 0.93-0.97, different voices 0.32-0.77. Erring high splits
/// one person into two, which renaming merges; erring low merges two people.
const SIMILARITY_THRESHOLD: f32 = 0.8;

/// Per-source audio of one VAD segment: 16kHz mono, sample-aligned.
#[derive(Debug, Clone)]
pub struct SourceAudio {
    pub mic: Vec<f32>,
    pub system: Vec<f32>,
}

/// Location of the speaker embedding model inside the app data directory.
pub fn model_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("models").join("speaker").join(MODEL_FILE)
}

/// Downloads the embedding model once; later recordings reuse the file.
/// The checksum matters: native model loading aborts the process on a bad file
/// (e.g. a captive portal page served with 200), so only verified bytes are kept.
pub async fn ensure_model(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    info!("Downloading speaker embedding model to {}", path.display());
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(180))
        .build()?;
    let bytes = client.get(MODEL_URL).send().await?.error_for_status()?.bytes().await?;
    let checksum = format!("{:x}", Sha256::digest(&bytes));
    if checksum != MODEL_SHA256 {
        anyhow::bail!("speaker model checksum mismatch ({} bytes, sha256 {})", bytes.len(), checksum);
    }
    let partial = path.with_extension("onnx.part");
    tokio::fs::write(&partial, &bytes).await?;
    tokio::fs::rename(&partial, path).await?;
    info!("Speaker embedding model ready ({} bytes)", bytes.len());
    Ok(())
}

struct Cluster {
    label: String,
    /// Sum of unit-length embeddings; its direction is the speaker centroid.
    sum: Vec<f32>,
}

/// Assigns a speaker label to each transcribed segment of one recording.
pub struct SpeakerTracker {
    model_path: PathBuf,
    extractor: Option<EmbeddingExtractor>,
    extractor_failed: bool,
    clusters: Vec<Cluster>,
    last_remote: Option<String>,
}

impl SpeakerTracker {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            extractor: None,
            extractor_failed: false,
            clusters: Vec::new(),
            last_remote: None,
        }
    }

    pub fn label(&mut self, audio: &SourceAudio) -> String {
        if is_mic_dominant(&audio.mic, &audio.system) {
            SPEAKER_ME.to_string()
        } else {
            self.label_voice(&audio.system)
        }
    }

    /// Labels by voice alone; also used for imported audio, where sources are mixed.
    pub fn label_voice(&mut self, system: &[f32]) -> String {
        if system.len() < MIN_EMBEDDING_SAMPLES {
            if let Some(last) = &self.last_remote {
                return last.clone();
            }
        }
        let Some(extractor) = self.extractor() else {
            return SPEAKER_OTHERS.to_string();
        };

        let tail = &system[system.len().saturating_sub(MAX_EMBEDDING_SAMPLES)..];
        let embedding = match extractor.compute_speaker_embedding(tail.to_vec(), SOURCE_SAMPLE_RATE) {
            Ok(embedding) => normalize(embedding),
            Err(e) => {
                warn!("Speaker embedding failed: {}", e);
                return self.last_remote.clone().unwrap_or_else(|| SPEAKER_OTHERS.to_string());
            }
        };

        let label = self.assign_cluster(embedding);
        self.last_remote = Some(label.clone());
        label
    }

    fn assign_cluster(&mut self, embedding: Vec<f32>) -> String {
        let best = self
            .clusters
            .iter_mut()
            .map(|c| (cosine(&c.sum, &embedding), c))
            .max_by(|a, b| a.0.total_cmp(&b.0));

        if let Some((similarity, cluster)) = best {
            if similarity >= SIMILARITY_THRESHOLD {
                for (s, e) in cluster.sum.iter_mut().zip(&embedding) {
                    *s += e;
                }
                return cluster.label.clone();
            }
        }

        let label = format!("Speaker {}", self.clusters.len() + 1);
        info!("New remote speaker detected: {}", label);
        self.clusters.push(Cluster { label: label.clone(), sum: embedding });
        label
    }

    /// Loads the extractor lazily: the model may still be downloading when recording starts.
    fn extractor(&mut self) -> Option<&mut EmbeddingExtractor> {
        if self.extractor.is_none() && !self.extractor_failed && self.model_path.exists() {
            let config = ExtractorConfig {
                model: self.model_path.to_string_lossy().into_owned(),
                provider: Some("cpu".to_string()),
                num_threads: Some(2),
                debug: false,
            };
            match EmbeddingExtractor::new(config) {
                Ok(extractor) => {
                    info!("Speaker embedding extractor loaded");
                    self.extractor = Some(extractor);
                }
                Err(e) => {
                    warn!("Speaker embedding extractor unavailable: {}", e);
                    self.extractor_failed = true;
                }
            }
        }
        self.extractor.as_mut()
    }
}

/// Frame-level vote: each active 30ms frame goes to the clearly louder source.
fn is_mic_dominant(mic: &[f32], system: &[f32]) -> bool {
    let (mut mic_frames, mut system_frames) = (0usize, 0usize);
    for (m, s) in mic.chunks(FRAME_SAMPLES).zip(system.chunks(FRAME_SAMPLES)) {
        let (m, s) = (rms(m), rms(s));
        if m.max(s) < SILENCE_RMS {
            continue;
        }
        if m > s * DOMINANCE_RATIO {
            mic_frames += 1;
        } else {
            system_frames += 1;
        }
    }
    mic_frames > system_frames
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(amplitude: f32, len: usize) -> Vec<f32> {
        (0..len).map(|i| amplitude * (i as f32 * 0.05).sin()).collect()
    }

    #[test]
    fn mic_only_speech_is_me() {
        let audio = SourceAudio { mic: tone(0.2, 16000), system: vec![0.0; 16000] };
        assert!(is_mic_dominant(&audio.mic, &audio.system));
    }

    #[test]
    fn system_speech_with_mic_bleed_is_remote() {
        let audio = SourceAudio { mic: tone(0.05, 16000), system: tone(0.2, 16000) };
        assert!(!is_mic_dominant(&audio.mic, &audio.system));
    }

    #[test]
    fn silence_is_not_me() {
        assert!(!is_mic_dominant(&vec![0.0; 16000], &vec![0.0; 16000]));
    }

    #[test]
    fn similar_embeddings_share_a_speaker() {
        let mut tracker = SpeakerTracker::new(PathBuf::from("/nonexistent"));
        let a = tracker.assign_cluster(normalize(vec![1.0, 0.1, 0.0]));
        let b = tracker.assign_cluster(normalize(vec![0.9, 0.2, 0.0]));
        let c = tracker.assign_cluster(normalize(vec![0.0, 0.0, 1.0]));
        assert_eq!(a, "Speaker 1");
        assert_eq!(b, "Speaker 1");
        assert_eq!(c, "Speaker 2");
    }

    #[test]
    fn remote_speech_without_model_is_others() {
        let mut tracker = SpeakerTracker::new(PathBuf::from("/nonexistent"));
        let audio = SourceAudio { mic: vec![0.0; 32000], system: tone(0.2, 32000) };
        assert_eq!(tracker.label(&audio), SPEAKER_OTHERS);
    }
}
