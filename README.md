# music-affective-dissonance

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![librosa](https://img.shields.io/badge/librosa-0.10+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)

Detecting affective dissonance in music using audio feature extraction and machine learning. Combines signal-processing descriptors inspired by [librosa](https://librosa.org/) with classical classifiers to identify tracks where the emotional valence of musical structure conflicts with the listener's expected affective state.

---

## What Is Affective Dissonance?

**Affective dissonance** in music refers to a mismatch between the emotional signal conveyed by the acoustic properties of a track and the listener's contextual or expected emotional state. It manifests as:

- A minor-key melody with energetic, upbeat rhythm (e.g., sad lyrics over a danceable beat)
- Bright timbral content (high spectral centroid) with low harmonic consonance
- High tempo with low valence score — tension without release
- Abrupt transitions in MFCC texture that violate continuity expectations

The phenomenon is studied in music information retrieval (MIR), affective computing, and music psychology. Detecting it automatically has applications in playlist curation, mood-aware recommendation systems, and therapeutic music selection.

---

## Methodology

1. **Feature extraction** — audio descriptors are computed per track, replicating the feature space produced by librosa's core analysis functions.
2. **Dissonance labelling** — tracks are labelled as dissonant when multiple conflicting affective dimensions co-occur (e.g., high energy + low valence + low harmonic stability).
3. **Classification** — SVM (RBF kernel) and Random Forest are trained on the feature space; performance is evaluated with confusion matrices and ROC-AUC.
4. **Interpretability** — feature importances identify which acoustic dimensions are most predictive of dissonance.

---

## Audio Features

| Feature | librosa Function | Description |
|---------|-----------------|-------------|
| `tempo` | `beat.beat_track` | Estimated BPM — drives perceived energy and urgency |
| `spectral_centroid` | `feature.spectral_centroid` | "Brightness" — higher = more high-frequency content |
| `mfcc_mean` | `feature.mfcc` | Timbral texture — averaged across 13 MFCC coefficients |
| `chroma_mean` | `feature.chroma_stft` | Harmonic content — averaged across 12 pitch classes |
| `energy` | `feature.rms` | Root-mean-square energy — overall loudness/intensity |
| `valence_score` | Composite | Inferred valence from chroma, mode, and tempo interaction |
| `zero_crossing_rate` | `feature.zero_crossing_rate` | Noisiness / percussiveness of the signal |
| `spectral_rolloff` | `feature.spectral_rolloff` | Frequency below which 85% of energy is concentrated |

---

## Model Approach

| Model | Notes |
|-------|-------|
| SVM (RBF kernel) | Effective in high-dimensional feature spaces; regularised with C=1.0 |
| Random Forest | 200 trees; captures non-linear feature interactions; provides importances |

Both models are evaluated on a held-out 20% test split with stratified sampling to preserve class balance.

---

## Setup

```bash
git clone https://github.com/arkaniton/music-affective-dissonance.git
cd music-affective-dissonance

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

jupyter notebook affective_dissonance_analysis.ipynb
```

---

## License

MIT — see [LICENSE](LICENSE)
