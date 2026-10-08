# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "soundfile>=0.12",
#   "numpy<2",
# ]
# ///
"""WO09 un-park, stage 0: fetch a small Hebrew audio sample.

FLEURS he_il (CC-BY-4.0, Google/CMU/Microsoft), train split, streamed —
20 clips saved as wav into the bucket for the WO09 probe. The probe
itself still fetches nothing (WO09 contract).
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

OUT = Path("/ckpt/audio-sample")
N = 20


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    ds = load_dataset("google/fleurs", "he_il", split="train", streaming=True)
    saved = 0
    for row in ds:
        if saved >= N:
            break
        audio = row["audio"]
        if len(audio["array"]) < 16000:  # skip <1s clips
            continue
        p = OUT / f"fleurs-he-{saved:03d}.wav"
        sf.write(str(p), audio["array"], audio["sampling_rate"])
        saved += 1
        if saved % 5 == 0:
            print(f"[fetch] {saved}/{N}", flush=True)
    print(f"[done] {saved} clips -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
