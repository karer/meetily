//! Offline speaker diarization for batch jobs (retranscription, import).
//!
//! The whole recording is available, so speakers are found globally instead of
//! segment by segment:
//! 1. voice embeddings (CAM++) on 1.5s windows with 0.75s hop over VAD speech
//! 2. spectral clustering on a subsample of windows; the speaker count comes from
//!    the eigengap unless the user gives it
//! 3. every window scored against the speaker centroids, then Viterbi smoothing
//!    so the label does not flicker mid-phrase
//! 4. VAD segments cut at speaker changes, so each transcribed piece has one speaker
//!
//! Parameters were tuned on a 3h, 3-speaker meeting against a word-level reference
//! (96% agreement); the sherpa-onnx built-in clustering scored 49% on the same audio.

use anyhow::Result;
use faer::{Mat, Side};
use sherpa_rs::speaker_id::EmbeddingExtractor;

use super::vad::SpeechSegment;

const SAMPLE_RATE: usize = 16000;
const WINDOW_SAMPLES: usize = SAMPLE_RATE * 3 / 2;
const HOP_SAMPLES: usize = SAMPLE_RATE * 3 / 4;
/// Shorter audio gives unreliable embeddings; such segments inherit a neighbour's speaker.
const MIN_WINDOW_SAMPLES: usize = SAMPLE_RATE / 2;
/// Spectral clustering runs on at most this many windows (dense eigen-decomposition);
/// matches the full-data result to 99.9% on the tuning meeting.
const MAX_LANDMARKS: usize = 3000;
const KNN: usize = 30;
pub const MAX_SPEAKERS: usize = 8;
const KMEANS_ITERATIONS: usize = 50;
/// Viterbi: emission = cosine similarity * EMISSION_SCALE; switching speaker costs
/// SWITCH_PENALTY inside continuous speech and a quarter of it across a pause.
const EMISSION_SCALE: f32 = 10.0;
const SWITCH_PENALTY: f32 = 3.0;
/// Cuts are moved to the quietest 100ms within this radius of the speaker change.
const CUT_SEARCH_SAMPLES: usize = SAMPLE_RATE * 2 / 5;
const ENERGY_WINDOW: usize = SAMPLE_RATE / 10;
/// Pieces shorter than this are not split off; they stay with the preceding speaker.
const MIN_PIECE_SAMPLES: usize = SAMPLE_RATE / 2;
/// VAD segments keep pauses of up to 2s; windows quieter than this fraction of the
/// median window RMS are left out, otherwise silence forms its own "speaker".
const QUIET_WINDOW_RATIO: f32 = 0.1;

struct Window {
    segment: usize,
    /// Offset into the segment's samples.
    start: usize,
    end: usize,
}

/// One VAD segment piece with a single speaker (0-based, in order of first appearance).
pub struct SpeakerPiece {
    pub segment: SpeechSegment,
    pub speaker: usize,
}

/// Splits VAD segments into single-speaker pieces.
/// `on_progress(done, total)` returns false to cancel.
pub fn diarize(
    extractor: &mut EmbeddingExtractor,
    segments: &[SpeechSegment],
    num_speakers: Option<usize>,
    mut on_progress: impl FnMut(usize, usize) -> bool,
) -> Result<Vec<SpeakerPiece>> {
    let windows = drop_quiet_windows(segments, make_windows(segments));
    let mut embeddings = Vec::with_capacity(windows.len());
    for (i, w) in windows.iter().enumerate() {
        if i % 100 == 0 && !on_progress(i, windows.len()) {
            anyhow::bail!("Diarization cancelled");
        }
        let samples = segments[w.segment].samples[w.start..w.end].to_vec();
        let embedding = extractor
            .compute_speaker_embedding(samples, SAMPLE_RATE as u32)
            .map_err(|e| anyhow::anyhow!("speaker embedding failed: {}", e))?;
        embeddings.push(normalize(embedding));
    }

    let labels = if embeddings.is_empty() {
        Vec::new()
    } else {
        let scores = speaker_scores(&embeddings, num_speakers);
        let pauses: Vec<bool> = windows
            .windows(2)
            .map(|p| p[0].segment != p[1].segment || p[1].start >= p[0].end)
            .collect();
        viterbi(&scores, &pauses)
    };
    let labels = order_by_first_appearance(&labels);
    Ok(split_segments(segments, &windows, &labels))
}

fn make_windows(segments: &[SpeechSegment]) -> Vec<Window> {
    let mut windows = Vec::new();
    for (segment, s) in segments.iter().enumerate() {
        let len = s.samples.len();
        if len < MIN_WINDOW_SAMPLES {
            continue;
        }
        if len <= WINDOW_SAMPLES {
            windows.push(Window { segment, start: 0, end: len });
            continue;
        }
        let mut start = 0;
        while start + WINDOW_SAMPLES < len + HOP_SAMPLES / 2 {
            windows.push(Window { segment, start, end: (start + WINDOW_SAMPLES).min(len) });
            start += HOP_SAMPLES;
        }
    }
    windows
}

/// Quiet windows get no label; their audio stays with the neighbouring speaker.
fn drop_quiet_windows(segments: &[SpeechSegment], windows: Vec<Window>) -> Vec<Window> {
    let rms: Vec<f32> = windows
        .iter()
        .map(|w| (energy(&segments[w.segment].samples[w.start..w.end]) / (w.end - w.start) as f32).sqrt())
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(f32::total_cmp);
    let Some(&median) = sorted.get(sorted.len() / 2) else {
        return windows;
    };
    windows
        .into_iter()
        .zip(rms)
        .filter(|&(_, r)| r >= median * QUIET_WINDOW_RATIO)
        .map(|(w, _)| w)
        .collect()
}

/// Cosine similarity of every window to every speaker centroid.
fn speaker_scores(embeddings: &[Vec<f32>], num_speakers: Option<usize>) -> Vec<Vec<f32>> {
    let step = embeddings.len().div_ceil(MAX_LANDMARKS);
    let landmarks: Vec<&Vec<f32>> = embeddings.iter().step_by(step).collect();
    let labels = spectral_clustering(&landmarks, num_speakers);
    let k = labels.iter().max().map_or(1, |m| m + 1);

    let dim = embeddings[0].len();
    let mut centroids = vec![vec![0.0f32; dim]; k];
    for (e, &l) in landmarks.iter().zip(&labels) {
        centroids[l].iter_mut().zip(e.iter()).for_each(|(c, x)| *c += x);
    }
    let centroids: Vec<Vec<f32>> = centroids.into_iter().map(normalize).collect();
    embeddings
        .iter()
        .map(|e| centroids.iter().map(|c| dot(c, e)).collect())
        .collect()
}

/// Normalized spectral clustering on a symmetrized kNN graph.
fn spectral_clustering(points: &[&Vec<f32>], num_speakers: Option<usize>) -> Vec<usize> {
    let n = points.len();
    if n < 3 {
        return vec![0; n];
    }
    let knn = KNN.min(n - 1);

    let mut affinity = vec![0.0f64; n * n];
    for i in 0..n {
        let mut sims: Vec<(f32, usize)> =
            (0..n).filter(|&j| j != i).map(|j| (dot(points[i], points[j]), j)).collect();
        sims.select_nth_unstable_by(knn - 1, |a, b| b.0.total_cmp(&a.0));
        for &(_, j) in &sims[..knn] {
            affinity[i * n + j] += 0.5;
            affinity[j * n + i] += 0.5;
        }
    }
    let inv_sqrt_degree: Vec<f64> = (0..n)
        .map(|i| {
            let d: f64 = affinity[i * n..(i + 1) * n].iter().sum();
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();
    let laplacian = Mat::<f64>::from_fn(n, n, |i, j| {
        let a = affinity[i * n + j] * inv_sqrt_degree[i] * inv_sqrt_degree[j];
        if i == j { 1.0 - a } else { -a }
    });
    let Ok(eigen) = laplacian.self_adjoint_eigen(Side::Lower) else {
        return vec![0; n];
    };
    let values: Vec<f64> = (0..n).map(|i| eigen.S()[i]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

    let k = match num_speakers {
        Some(k) => k.clamp(1, n),
        None => eigengap(&order.iter().map(|&i| values[i]).collect::<Vec<_>>()),
    };
    let u = eigen.U();
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|r| normalize(order[..k].iter().map(|&c| u[(r, c)] as f32).collect()))
        .collect();
    kmeans(&rows, k)
}

/// Speaker count = position of the largest gap among the smallest eigenvalues.
fn eigengap(sorted_values: &[f64]) -> usize {
    let limit = (MAX_SPEAKERS + 1).min(sorted_values.len());
    (1..limit)
        .max_by(|&a, &b| {
            let gap = |i: usize| sorted_values[i] - sorted_values[i - 1];
            gap(a).total_cmp(&gap(b))
        })
        .unwrap_or(1)
}

/// Spherical k-means with deterministic farthest-point initialisation.
fn kmeans(points: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut centers = vec![points[0].clone()];
    while centers.len() < k {
        let farthest = points
            .iter()
            .min_by(|a, b| {
                let closest = |p: &Vec<f32>| centers.iter().map(|c| dot(c, p)).fold(f32::MIN, f32::max);
                closest(a).total_cmp(&closest(b))
            })
            .unwrap();
        centers.push(farthest.clone());
    }
    let mut labels = vec![0; points.len()];
    for _ in 0..KMEANS_ITERATIONS {
        for (l, p) in labels.iter_mut().zip(points) {
            *l = argmax(&centers.iter().map(|c| dot(c, p)).collect::<Vec<_>>());
        }
        for (j, center) in centers.iter_mut().enumerate() {
            let mut sum = vec![0.0f32; center.len()];
            let mut members = 0;
            for (p, &l) in points.iter().zip(&labels) {
                if l == j {
                    sum.iter_mut().zip(p).for_each(|(s, x)| *s += x);
                    members += 1;
                }
            }
            if members > 0 {
                *center = normalize(sum);
            }
        }
    }
    labels
}

/// Most likely speaker sequence; `pauses[i]` is true when window i+1 does not overlap window i.
fn viterbi(scores: &[Vec<f32>], pauses: &[bool]) -> Vec<usize> {
    let k = scores[0].len();
    let mut score: Vec<f32> = scores[0].iter().map(|s| s * EMISSION_SCALE).collect();
    let mut back = vec![vec![0usize; k]; scores.len()];
    for i in 1..scores.len() {
        let penalty = if pauses[i - 1] { SWITCH_PENALTY / 4.0 } else { SWITCH_PENALTY };
        let prev = score.clone();
        for j in 0..k {
            let (from, best) = (0..k)
                .map(|f| (f, prev[f] - if f == j { 0.0 } else { penalty }))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            back[i][j] = from;
            score[j] = best + scores[i][j] * EMISSION_SCALE;
        }
    }
    let mut path = vec![argmax(&score); scores.len()];
    for i in (1..scores.len()).rev() {
        path[i - 1] = back[i][path[i]];
    }
    path
}

fn order_by_first_appearance(labels: &[usize]) -> Vec<usize> {
    let mut mapping: Vec<Option<usize>> = vec![None; MAX_SPEAKERS.max(labels.iter().max().map_or(0, |m| m + 1))];
    let mut next = 0;
    labels
        .iter()
        .map(|&l| {
            *mapping[l].get_or_insert_with(|| {
                next += 1;
                next - 1
            })
        })
        .collect()
}

/// Cuts each segment where its windows change speaker. Segments without windows
/// (too short) take the speaker of the previous piece.
fn split_segments(segments: &[SpeechSegment], windows: &[Window], labels: &[usize]) -> Vec<SpeakerPiece> {
    let mut pieces: Vec<SpeakerPiece> = Vec::new();
    let mut w = 0;
    for (index, segment) in segments.iter().enumerate() {
        // (cut position in samples, speaker from there on)
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut prev: Option<(&Window, usize)> = None;
        while w < windows.len() && windows[w].segment == index {
            let label = labels[w];
            match prev {
                None => runs.push((0, label)),
                Some((pw, pl)) if pl != label => {
                    let change = (pw.start + pw.end + windows[w].start + windows[w].end) / 4;
                    let cut = quietest_point(&segment.samples, change).max(runs.last().map_or(0, |r| r.0));
                    runs.push((cut, label));
                }
                _ => {}
            }
            prev = Some((&windows[w], label));
            w += 1;
        }
        if runs.is_empty() {
            let speaker = pieces.last().map_or(0, |p| p.speaker);
            runs.push((0, speaker));
        }

        let ms_per_sample = (segment.end_timestamp_ms - segment.start_timestamp_ms) / segment.samples.len().max(1) as f64;
        // Too short or same speaker: absorbed by the previous piece
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        for (i, &(start, speaker)) in runs.iter().enumerate() {
            let end = runs.get(i + 1).map_or(segment.samples.len(), |r| r.0);
            let absorbed = bounds.last().is_some_and(|last| last.1 == speaker || end - start < MIN_PIECE_SAMPLES);
            if !absorbed {
                bounds.push((start, speaker));
            }
        }
        for (i, &(start, speaker)) in bounds.iter().enumerate() {
            let end = bounds.get(i + 1).map_or(segment.samples.len(), |b| b.0);
            pieces.push(SpeakerPiece {
                segment: SpeechSegment {
                    samples: segment.samples[start..end].to_vec(),
                    start_timestamp_ms: segment.start_timestamp_ms + start as f64 * ms_per_sample,
                    end_timestamp_ms: segment.start_timestamp_ms + end as f64 * ms_per_sample,
                    confidence: segment.confidence,
                },
                speaker,
            });
        }
    }
    pieces
}

/// Centre of the quietest 100ms window within CUT_SEARCH_SAMPLES of `around`.
fn quietest_point(samples: &[f32], around: usize) -> usize {
    let lo = around.saturating_sub(CUT_SEARCH_SAMPLES);
    let hi = (around + CUT_SEARCH_SAMPLES).min(samples.len().saturating_sub(ENERGY_WINDOW));
    (lo..hi.max(lo))
        .step_by(160)
        .min_by(|&a, &b| energy(&samples[a..a + ENERGY_WINDOW]).total_cmp(&energy(&samples[b..b + ENERGY_WINDOW])))
        .map_or(around.min(samples.len()), |s| s + ENERGY_WINDOW / 2)
}

fn energy(samples: &[f32]) -> f32 {
    samples.iter().map(|x| x * x).sum()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vector near `axis` in `dim` dimensions; `wobble` varies it deterministically.
    fn voice(axis: usize, dim: usize, wobble: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v[(axis + 1 + wobble % 3) % dim] += 0.25;
        normalize(v)
    }

    #[test]
    fn spectral_clustering_finds_speaker_count() {
        let points: Vec<Vec<f32>> = (0..120).map(|i| voice(i % 3 * 4, 12, i)).collect();
        let refs: Vec<&Vec<f32>> = points.iter().collect();
        let labels = spectral_clustering(&refs, None);
        assert_eq!(labels.iter().max(), Some(&2), "expected 3 speakers");
        for i in 0..120 {
            assert_eq!(labels[i], labels[i % 3], "point {} not with its speaker", i);
        }
    }

    #[test]
    fn spectral_clustering_respects_given_count() {
        let points: Vec<Vec<f32>> = (0..120).map(|i| voice(i % 3 * 4, 12, i)).collect();
        let refs: Vec<&Vec<f32>> = points.iter().collect();
        assert_eq!(spectral_clustering(&refs, Some(2)).iter().max(), Some(&1));
    }

    #[test]
    fn viterbi_ignores_single_window_blip() {
        let a = vec![0.9, 0.5];
        let b = vec![0.5, 0.9];
        let scores = vec![a.clone(), a.clone(), b.clone(), a.clone(), a, b.clone(), b.clone(), b];
        let pauses = vec![false; scores.len() - 1];
        assert_eq!(viterbi(&scores, &pauses), vec![0, 0, 0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn labels_follow_first_appearance() {
        assert_eq!(order_by_first_appearance(&[2, 2, 0, 1, 0]), vec![0, 0, 1, 2, 1]);
    }

    fn segment(seconds: usize, start_ms: f64) -> SpeechSegment {
        let samples = vec![0.1f32; seconds * SAMPLE_RATE];
        SpeechSegment {
            end_timestamp_ms: start_ms + seconds as f64 * 1000.0,
            samples,
            start_timestamp_ms: start_ms,
            confidence: 1.0,
        }
    }

    #[test]
    fn segments_split_at_speaker_change() {
        let segments = vec![segment(6, 1000.0)];
        let windows = make_windows(&segments);
        let half = windows.len() / 2;
        let labels: Vec<usize> = (0..windows.len()).map(|i| usize::from(i >= half)).collect();
        let pieces = split_segments(&segments, &windows, &labels);
        assert_eq!(pieces.len(), 2);
        assert_eq!((pieces[0].speaker, pieces[1].speaker), (0, 1));
        assert_eq!(pieces[0].segment.start_timestamp_ms, 1000.0);
        assert_eq!(pieces[1].segment.end_timestamp_ms, 7000.0);
        assert_eq!(pieces[0].segment.end_timestamp_ms, pieces[1].segment.start_timestamp_ms);
        assert_eq!(pieces[0].segment.samples.len() + pieces[1].segment.samples.len(), 6 * SAMPLE_RATE);
    }

    #[test]
    fn quiet_windows_are_dropped() {
        let mut loud = segment(6, 0.0);
        loud.samples[3 * SAMPLE_RATE..].iter_mut().for_each(|x| *x = 0.001);
        let segments = vec![loud];
        let windows = drop_quiet_windows(&segments, make_windows(&segments));
        assert!(!windows.is_empty());
        assert!(windows.iter().all(|w| w.start < 3 * SAMPLE_RATE), "quiet tail must not be embedded");
    }

    #[test]
    fn short_segment_inherits_previous_speaker() {
        let segments = vec![segment(3, 0.0), SpeechSegment { samples: vec![0.1; 1000], ..segment(0, 4000.0) }];
        let windows = make_windows(&segments);
        let labels = vec![1; windows.len()];
        let pieces = split_segments(&segments, &windows, &labels);
        assert_eq!(pieces.iter().map(|p| p.speaker).collect::<Vec<_>>(), vec![1, 1]);
    }

    #[test]
    fn ui_speaker_limit_matches() {
        let ui = include_str!("../../../src/components/SpeakerCountSelect.tsx");
        assert!(
            ui.contains(&format!("export const MAX_SPEAKERS = {};", MAX_SPEAKERS)),
            "SpeakerCountSelect.tsx MAX_SPEAKERS differs from diarization::MAX_SPEAKERS"
        );
    }

    /// Acceptance run on a real recording. Writes the pieces as JSON for comparison
    /// with a reference transcript:
    /// DIARIZE_AUDIO=<raw f32le 16kHz mono> DIARIZE_MODEL=<CAM++ onnx> DIARIZE_OUT=<json>
    /// cargo test --release diarize_real_recording -- --ignored --nocapture
    #[test]
    #[ignore]
    fn diarize_real_recording() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{} not set", k));
        let bytes = std::fs::read(var("DIARIZE_AUDIO")).unwrap();
        let audio: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let started = std::time::Instant::now();
        let segments = crate::audio::vad::get_speech_chunks_with_progress(
            &audio,
            crate::audio::retranscription::VAD_REDEMPTION_TIME_MS,
            |_, _| true,
        )
        .unwrap();
        println!("VAD: {} segments in {:?}", segments.len(), started.elapsed());
        let mut extractor =
            crate::audio::speaker_tracker::new_extractor(std::path::Path::new(&var("DIARIZE_MODEL")), 4).unwrap();
        let started = std::time::Instant::now();
        let pieces = diarize(&mut extractor, &segments, None, |_, _| true).unwrap();
        println!("Diarization: {} pieces in {:?}", pieces.len(), started.elapsed());
        let json: Vec<_> = pieces
            .iter()
            .map(|p| serde_json::json!({"start": p.segment.start_timestamp_ms / 1000.0, "end": p.segment.end_timestamp_ms / 1000.0, "speaker": p.speaker}))
            .collect();
        std::fs::write(var("DIARIZE_OUT"), serde_json::to_string(&json).unwrap()).unwrap();
    }
}
