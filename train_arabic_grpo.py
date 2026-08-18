"""GRPO on the Arabic ByT5 diacritizer — gold-haraqat reward, no teacher.

Why GRPO after RAFT: rejection sampling (RAFT) only reinforces winners;
when the posterior is split between legal readings (our diagnosed
residual — 98.7% of errors are phonotactically legal), positive-only
updates can't sharpen the choice. GRPO adds the negative gradient: per
prompt, sample G candidates, reward = -letter-aligned DER vs gold
(deterministic oracle), advantage = group-normalized reward, update all
G samples weighted by advantage plus a KL leash to the reference policy.

No LLM teacher anywhere in the loop (standing rule): the only reward
source is gold haraqat.

Init preference: run-005-context best (paragraph-context) if present,
else run-003-domain best. Selection on the frozen private dev; the
public benchmark is measured once at the end (windowed zero-skip at the
model's training context).

Usage:
    modal run --detach train_arabic_grpo.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_grpo/run-001"
N_VAL = 2_000
PROMPT_POOL = 100_000
STEPS = 400
GROUP = 8
PROMPTS_PER_STEP = 1
GRAD_ACCUM = 16
TEMP = 1.0
LR = 1e-5
KL_BETA = 0.05
MAX_BYTES = 1400
DEV_N = 500
EVAL_EVERY = 100
SAVE_EVERY = 50

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "pandas", "pyarrow", "tqdm")
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-grpo", image=image)


def letter_haraqat(text: str) -> list[list[str]]:
    seq: list[list[str]] = []
    for ch in text:
        if DIACRITICS_RE.match(ch):
            if seq:
                seq[-1][1] += ch
        else:
            seq.append([ch, ""])
    return seq


def der(pred: str, gold: str) -> float:
    # Graded, alignment-based: binary letter-mismatch penalty would make
    # every temp-1.0 sample score 1.0 (one wrong byte = full penalty),
    # collapsing group advantages to zero. Align letters instead and
    # count haraqat errors on gold-vocalized positions.
    from difflib import SequenceMatcher

    p, g = letter_haraqat(pred), letter_haraqat(gold)
    pl, gl = [a[0] for a in p], [b[0] for b in g]
    sm = SequenceMatcher(None, gl, pl, autojunk=False)
    err = tot = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                if gl[i1 + k] == " ":
                    continue
                if g[i1 + k][1]:
                    tot += 1
                    err += p[j1 + k][1] != g[i1 + k][1]
                elif p[j1 + k][1]:
                    tot += 1
                    err += 1
        else:
            for k in range(i1, i2):
                if gl[k] != " " and g[k][1]:
                    tot += 1
                    err += 1
    return err / tot if tot else 1.0


def load_pool() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    lines = [
        l.strip()
        for l in Path("/datasets/sadeed-decontam/train.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    random.Random(42).shuffle(lines)

    def valid(line: str) -> tuple[str, str] | None:
        src = DIACRITICS_RE.sub("", line)
        if not src:
            return None
        if len(src.encode("utf-8")) > MAX_BYTES or len(line.encode("utf-8")) > MAX_BYTES:
            return None
        return src, line

    dev = [p for p in (valid(l) for l in lines[:N_VAL]) if p][:DEV_N]
    pool = [p for p in (valid(l) for l in lines[N_VAL : N_VAL + PROMPT_POOL]) if p]
    random.Random(43).shuffle(pool)
    return dev, pool


@app.function(
    gpu="A100",
    timeout=11 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def run() -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoints_volume.reload()
    run_dir = Path("/checkpoints") / RUN
    if (run_dir / "EVAL_DONE").exists():
        return {"status": "already-done"}
    run_dir.mkdir(parents=True, exist_ok=True)

    r5 = Path("/checkpoints/rababa_arabic_byt5/run-005-context/best")
    init_dir = str(r5) if r5.is_dir() else "/checkpoints/rababa_arabic_byt5/run-003-domain/best"
    print(f"[init] {init_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(init_dir)
    try:
        policy = AutoModelForSeq2SeqLM.from_pretrained(
            init_dir, attn_implementation="sdpa").to("cuda")
    except Exception:
        policy = AutoModelForSeq2SeqLM.from_pretrained(init_dir).to("cuda")
    try:
        ref = AutoModelForSeq2SeqLM.from_pretrained(
            init_dir, attn_implementation="sdpa").to("cuda").eval()
    except Exception:
        ref = AutoModelForSeq2SeqLM.from_pretrained(init_dir).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    device = next(policy.parameters()).device
    # byt5-base activations for 16x(1400+2800) byte-tokens with grad exceed
    # an A100; checkpointing trades ~30% speed for ~10x activation memory.
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01)

    dev, pool = load_pool()
    print(f"[data] dev={len(dev)} pool={len(pool)}", flush=True)

    def greedy_der(pairs: list[tuple[str, str]]) -> float:
        ders = []
        policy.eval()
        with torch.no_grad():
            for i in range(0, len(pairs), 16):
                batch = [s for s, _ in pairs[i : i + 16]]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=MAX_BYTES).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = policy.generate(**enc, max_new_tokens=2 * MAX_BYTES, num_beams=1)
                outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
                ders.extend(der(o, g) for o, (_, g) in zip(outs, pairs[i : i + 16]))
        return sum(ders) / len(ders)

    def token_logprobs(model, src_texts, tgt_texts):
        """Per-sample summed logprob and token count (teacher forcing)."""
        enc = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_BYTES).to(device)
        labels = tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2 * MAX_BYTES)
        lab = labels["input_ids"].to(device)
        attn = labels["attention_mask"].to(device)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           labels=lab).logits
        probs = torch.softmax(logits.float(), dim=-1)
        logprobs = probs.log()
        tgt = lab[:, 1:].unsqueeze(-1)
        lp = torch.gather(logprobs[:, :-1], 2, tgt).squeeze(-1)
        mask = attn[:, 1:].float()
        # GTPO-style dynamic entropy weighting (arXiv 2508.04349): credit
        # concentrates on high-entropy decision tokens (the haraqat), not the
        # copied letters. Weights are redistribution constants — detached so
        # no gradient flows through the entropy term itself.
        with torch.no_grad():
            ent = -(probs[:, :-1] * logprobs[:, :-1]).sum(dim=-1)
            w = (ent / ent.mean().clamp(min=1e-6)).clamp(0.1, 4.0) * mask
        return (lp * w).sum(dim=1), w.sum(dim=1)

    state_path = run_dir / "state.json"
    metrics_path = run_dir / "metrics.jsonl"
    start_step = 0
    best_dev = None
    if state_path.exists():
        st = json.loads(state_path.read_text())
        start_step = st["step"]
        best_dev = st.get("best_dev")
        ckpt = run_dir / "policy_last"
        if (ckpt).is_dir():
            policy.load_state_dict(
                AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).state_dict())
            if (ckpt / "optimizer.pt").exists():
                opt.load_state_dict(torch.load(str(ckpt / "optimizer.pt"), map_location="cpu"))
            print(f"[resume] step {start_step}", flush=True)

    if best_dev is None:
        best_dev = greedy_der(dev)
        print(f"[dev] init DER={best_dev:.4%}", flush=True)

    rng = random.Random(7)
    ptr = 0
    policy.train()
    for step in range(start_step, STEPS):
        for _ in range(GRAD_ACCUM):
            batch_pairs = []
            while len(batch_pairs) < PROMPTS_PER_STEP:
                batch_pairs.append(pool[ptr % len(pool)])
                ptr += 1
            srcs = [s for s, _ in batch_pairs]
            golds = [g for _, g in batch_pairs]

            enc = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True,
                            max_length=MAX_BYTES).to(device)
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    gen = policy.generate(
                        **enc, max_new_tokens=2 * MAX_BYTES, num_beams=1,
                        do_sample=True, temperature=TEMP, num_return_sequences=GROUP,
                    )
            outs = tokenizer.batch_decode(gen, skip_special_tokens=True)

            rewards = []
            for gi in range(len(srcs)):
                for k in range(GROUP):
                    rewards.append(-der(outs[gi * GROUP + k], golds[gi]))
            import statistics
            mu = statistics.mean(rewards)
            sd = statistics.pstdev(rewards) + 1e-4
            advs = [(r - mu) / sd for r in rewards]

            rep_srcs = [s for s in srcs for _ in range(GROUP)]
            lp_pol, n_tok = token_logprobs(policy, rep_srcs, outs)
            with torch.no_grad():
                lp_ref, _ = token_logprobs(ref, rep_srcs, outs)
            kl = (lp_pol - lp_ref).clamp(min=-10, max=10)
            A = torch.tensor(advs, device=device, dtype=torch.float32)
            loss = -((A / n_tok.clamp(min=1)) * lp_pol).mean() + KL_BETA * kl.mean()
            (loss / GRAD_ACCUM).backward()
            if step % 50 == 0:
                print(f"[step {step}] loss={loss.item():.4f} mean_r={mu:.4f} best={best_dev:.4%}", flush=True)

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        policy.train()

        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            d = greedy_der(dev)
            print(f"[dev] step {step+1}: DER={d:.4%} (best={best_dev:.4%})", flush=True)
            if d < best_dev:
                best_dev = d
                best_dir = run_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step + 1, "dev_der": d, "best_dev": best_dev}) + "\n")
            checkpoints_volume.commit()

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
            last = run_dir / "policy_last"
            last.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(last))
            tokenizer.save_pretrained(str(last))
            torch.save(opt.state_dict(), str(last / "optimizer.pt"))
            state_path.write_text(json.dumps({"step": step + 1, "best_dev": best_dev}))
            checkpoints_volume.commit()

    # ---- final one-shot benchmark from best ----
    import pandas as pd
    import pyarrow.parquet as pq
    from difflib import SequenceMatcher

    eval_dir = run_dir / "best"
    eval_model = AutoModelForSeq2SeqLM.from_pretrained(str(eval_dir)).to(device)
    eval_model.eval()

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    def split_windows(text: str, budget: int = MAX_BYTES) -> list[str]:
        if len(text.encode("utf-8")) <= budget:
            return [text]
        words = text.split()
        wins, cur, n = [], [], 0
        for w in words:
            c = len(w.encode("utf-8")) + 1
            if cur and n + c > budget:
                wins.append(" ".join(cur))
                cur, n = [], 0
            cur.append(w)
            n += c
        if cur:
            wins.append(" ".join(cur))
        return wins

    def project_haraqat(pred: str, text: str) -> str:
        pred_haraqat = [""]
        for ch in pred:
            if DIACRITICS_RE.match(ch):
                pred_haraqat[-1] += ch
            else:
                pred_haraqat.append("")
        pred_haraqat = pred_haraqat[1:]
        pred_letters = [c for c in pred if not DIACRITICS_RE.match(c)]
        text_letters = [c for c in text if not DIACRITICS_RE.match(c)]
        sm = SequenceMatcher(None, text_letters, pred_letters, autojunk=False)
        out = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    out.append(text_letters[i1 + k] + pred_haraqat[j1 + k])
            else:
                for k in range(i1, i2):
                    out.append(text_letters[k])
        return "".join(out)

    all_windows: list[str] = []
    counts: list[int] = []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)

    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(all_windows), 8):
            batch = all_windows[i : i + 8]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=MAX_BYTES).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = eval_model.generate(**enc, max_new_tokens=2 * MAX_BYTES, num_beams=1)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            if (i // 8) % 40 == 0:
                print(f"[gen] {i + len(batch)}/{len(all_windows)}", flush=True)

    k = 0
    paragraphs = []
    for text, c in zip(inputs, counts):
        stitched = " ".join(preds[k : k + c])
        k += c
        paragraphs.append(project_haraqat(stitched, text))

    csv_path = Path("/tmp/sadeed_grpo_windowed.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (run_dir / "sadeed_preds_windowed.csv").write_text(csv_path.read_text(), encoding="utf-8")

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("\n===== GRPO windowed zero-skip =====", flush=True)
    E.report_errors_on_csv_file(
        str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
        gt_missing_diacritic_is_error=False)

    (run_dir / "EVAL_DONE").touch()
    checkpoints_volume.commit()
    return {"run": RUN, "best_dev": best_dev}


@app.local_entrypoint()
def main():
    run.remote()
