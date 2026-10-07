# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "librosa>=0.10",
#   "numpy<2",
# ]
# ///
"""WO09 (PARKED probe): Hebrew audio -> phoneme pseudo-label viability.

Stage 1: universal phoneme CTC ASR (facebook/wav2vec2-lv-60-espeak-cv-ft)
over a small local dir of Hebrew audio (the script fetches NOTHING).
Stage 2: forced-alignment viability — espeak-style reference phonemes of
our plane-restored nikud text vs the ASR phoneme stream; token agreement
stats + per-hour labeling cost estimate. Output: a go/no-go report for a
2027 audio-supervision campaign (the ReNikud direction, arXiv 2606.20179).

Usage:
    hf jobs uv run --flavor l4x1 --timeout 1h \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      probe_hebrew_audio_pseudolabel.py --audio-dir /ckpt/audio-sample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ASR = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="probe_report.json")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    clips = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in (".wav", ".flac", ".mp3"))[:args.limit]
    if not clips:
        print(f"no audio clips under {audio_dir} — probe requires local audio "
              f"(parked WO; nothing is fetched).", flush=True)
        return

    import librosa
    import torch
    from transformers import AutoProcessor, AutoModelForCTC

    processor = AutoProcessor.from_pretrained(ASR)
    model = AutoModelForCTC.from_pretrained(ASR).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    results = []
    t0 = time.time()
    for clip in clips:
        speech, _ = librosa.load(str(clip), sr=16000, mono=True)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = logits.argmax(-1)
        phon = processor.batch_decode(ids)
        results.append({"clip": clip.name, "phonemes": phon[0],
                        "seconds": round(len(speech) / 16000, 2)})

    wall = time.time() - t0
    total_s = sum(r["seconds"] for r in results) or 1.0
    report = {
        "status": "POC: ASR stage runs; alignment stage NOT implemented (parked)",
        "clips": len(results),
        "audio_seconds": total_s,
        "wall_s": round(wall, 1),
        "rtf_asr_only": round(wall / total_s, 3),
        "hourly_label_cost_usd": round(2.50 * wall / total_s, 2),
        "next_steps": "espeak reference phonemization of plane-restored text; "
                      "CTC forced alignment; agreement stats; 2027 campaign gate",
        "samples": results[:3],
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"},
                     ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
