"""The inference configuration measured best on the held-out test split.

A grid ablation run on 2026-08-17 scored ten configurations over the same 51
held-out songs, measuring how accurately each places the *first chorus start* --
the timestamp an incoming track is aligned to when mixing. The full table is in
``data/trials.csv``; the five rows that matter:

===================================  =======  ============  ==================
Configuration                          Exact  Within 1 bar  Median error
===================================  =======  ============  ==================
Extrapolated grid, threshold+smooth     7.8%         43.1%   5.9 beats
Extrapolated grid, Viterbi             11.8%         45.1%   5.6 beats
Extrapolated grid, Viterbi, snapping   58.8%         64.7%   2.0 beats
Tracked downbeats, Viterbi             58.8%         70.6%   3.0 beats
Tracked downbeats, Viterbi, snapping   76.5%         80.4%   1.0 beat
===================================  =======  ============  ==================

The last row is what these constants encode. Two supporting findings explain
the gap: the extrapolated grid sits about one beat out of phase, because it
anchors on the first beat librosa detects and assumes that beat is a downbeat;
and frame F1 does not discriminate at all (every configuration scores 0.869 to
0.905), because all of them find similar *amounts* of chorus and differ only in
where the boundaries land.

GRID_SOURCE and MODEL_URL must change together. The shipped checkpoint was
trained on the tracked-downbeat grid, so reading it with the extrapolated grid
is a train/inference mismatch and none of the numbers above hold.
"""

# Bar grid: "beat_this" takes bar lines from downbeats tracked by Beat This!
# (ISMIR 2024); "librosa" extrapolates one tempo over the whole song.
GRID_SOURCE = "beat_this"

# Decoder turning per-bar chorus probabilities into segments: "viterbi" runs
# the two-state HMM in pytorch_core.decoding; "smooth" thresholds at 0.5.
DECODE = "viterbi"

# Cost of switching between chorus and non-chorus at a bar line.
VITERBI_SWITCH_PENALTY = 2.0

# Decoded runs shorter than this many bars are dissolved.
VITERBI_MIN_BARS = 4

# Move each boundary to a nearby tracked downbeat after decoding.
SNAP_DOWNBEATS = True

# Half-width, in bars, of the window searched for that downbeat.
SNAP_WINDOW_BARS = 2.0
