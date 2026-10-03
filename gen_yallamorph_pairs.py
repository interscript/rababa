"""Generate the r9 MORPH aux corpus from CamelMorph MSA (CAMeL Tools).

Pipeline:
1. Load the YallaMorph lemma sample (4,795 lemmas; xlsx sample files).
2. Enumerate feature specs per POS (baseword + clitic slots).
3. Generate diacritized forms with camel_tools Generator.
4. Validate against YallaMorph few-shot gold examples (prompt files).
5. Emit build_lines output (src<TAB>target) + stats.

Usage:
    /tmp/camelenv/bin/python gen_yallamorph_pairs.py \
        --yalla-dir /tmp/yallamorph --out data/yallamorph-aux/lines.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from build_yallamorph_aux import (
    aux_pair,
    build_lines,
    iter_adj_specs,
    iter_noun_specs,
    iter_verb_specs,
)

# YallaMorph prompt vocabulary -> calima-msa-r13 generation-DB feature codes
Y2C = {
    "pos": {"verb": "verb", "noun": "noun", "adjective": "adj", "adj": "adj",
            "v": "verb", "n": "noun"},
    "per": {"1st": "1", "2nd": "2", "3rd": "3", "1": "1", "2": "2", "3": "3"},
    "gen": {"m": "m", "f": "f", "Masculine": "m", "Feminine": "f"},
    "num": {"s": "s", "d": "d", "p": "p", "Singular": "s", "Dual": "d", "Plural": "p"},
    "asp": {"pv": "p", "iv": "i", "c": "c", "Perfective": "p", "Imperfective": "i", "Command": "c"},
    "mod": {"i": "i", "s": "s", "j": "j", "el": "e", "eh": "e",
            "Indicative": "i", "Subjunctive": "s", "Jussive": "j",
            "ENERGETIC_LIGHT": "e", "ENERGETIC_HEAVY": "e"},
    "voz": {"a": "a", "p": "p", "Active": "a", "Passive": "p"},
    "cas": {"nom": "n", "acc": "a", "gen": "g",
            "Nominative": "n", "Accusative": "a", "Genitive": "g"},
    "stt": {"d": "d", "i": "i", "c": "c", "definite": "d", "indefinite": "i", "construct": "c"},
}

# clitic slots (YallaMorph name -> calima slot); values = calima-msa-r13 vocabulary
PROC_SLOTS = {
    "proclitics_0": "prc0",
    "proclitics_1": "prc1",
    "proclitics_2": "prc2",
    "proclitics_3": "prc3",
}
PROC_VALUES = {
    "proclitics_0": ("0", "Al_det"),
    "proclitics_1": ("0", "bi_prep", "li_prep", "ka_prep", "fiy_prep", "la_prep"),
    "proclitics_2": ("0", "wa_conj", "fa_conj"),
    "proclitics_3": ("0", ">a_ques"),
}
ENC_VALUES = ("0", "1s_poss", "2ms_poss", "2fs_poss", "3ms_poss", "3fs_poss", "1p_poss", "3mp_poss", "3fp_poss")


def calima_spec(spec: dict) -> dict:
    out = {}
    for k, v in spec.items():
        k = "vox" if k == "voz" else k
        if k in Y2C:
            if v in Y2C[k]:
                out[k] = Y2C[k][v]
        else:
            out[k] = v
    return out


# Camel Morph MSA v1.0 (LREC-COLING 2024 release, CC BY 4.0) — the
# resource YallaMorph was constructed from; usable directly by
# camel_tools MorphologyDB. Fetch:
#   git clone https://github.com/CAMeL-Lab/camel_morph
DB_PATH = ("/tmp/camel_morph/official_releases/lrec-coling2024_release"
           "/databases/camel-morph-msa/camel_morph_msa_v1.0.db")


def make_generator(db_path: str = DB_PATH):
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.generator import Generator

    return Generator(MorphologyDB(db_path, flags="g"))


def iter_noun_clitic_specs() -> list[dict]:
    """Bounded clitic variants — the full cross product explodes (14 extra
    per base) and would overrun the 300k line cap 20x over."""
    base = list(iter_noun_specs()) + list(iter_adj_specs())
    enc = ENC_VALUES[1:]
    specs = []
    for i, s in enumerate(base):
        specs.append(s)
        specs.append({**s, "prc0": "Al_det"})
        specs.append({**s, "prc1": "bi_prep"})
        specs.append({**s, "prc0": "Al_det", "prc2": "wa_conj"})
        specs.append({**s, "enc0": enc[i % len(enc)]})
    return specs


def iter_verb_clitic_specs() -> list[dict]:
    base = [s for s in iter_verb_specs()]
    specs = list(base)
    for s in base:
        if s["asp"] == "iv":
            for p1 in ("bi_prep", "la_prep", "sa_fut"):
                specs.append({**s, "prc1": p1})
        for p2 in ("wa_conj", "fa_conj"):
            specs.append({**s, "prc2": p2})
    return specs


def load_lemmas(yalla_dir: Path) -> dict[str, set[str]]:
    import openpyxl

    pos_files = {
        "v": ["Base Sample/Verb", "Clitics Sample/Verb"],
        "n": ["Base Sample/Noun", "Clitics Sample/Noun"],
        "adj": ["Base Sample/Adj", "Clitics Sample/Adj"],
    }
    lemmas: dict[str, set[str]] = {"v": set(), "n": set(), "adj": set()}
    for pos, subdirs in pos_files.items():
        for sub in subdirs:
            d = yalla_dir / "YallaMorph_Data" / sub
            if not d.exists():
                continue
            for x in sorted(d.glob("*.xlsx")):
                ws = openpyxl.load_workbook(x, read_only=True).active
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row and row[0]:
                        lemmas[pos].add(str(row[0]).strip())
    return lemmas


def parse_fewshot_gold(yalla_dir: Path) -> list[dict]:
    """(input feature dict, set of gold forms) from the few-shot prompts."""
    gold = []
    prompts = yalla_dir / "Prompts" / "English_prompts" / "Few_shots"
    for p in sorted(prompts.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"Input:\n(.*?)\nOutput:\n(.*?)\n\n", text, re.S):
            feat, out = {}, {}
            for line in m.group(1).strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    feat[k.strip()] = v.strip()
            try:
                out = json.loads(m.group(2).strip())
            except json.JSONDecodeError:
                continue
            if out.get("status") == "OK" and out.get("arabic_forms"):
                gold.append({"file": p.name, "feat": feat, "forms": set(out["arabic_forms"])})
    return gold


_GEN = None


def _get_generator():
    global _GEN
    if _GEN is None:
        _GEN = make_generator()
    return _GEN


# Underspecified generation is the correct interface: the DB's merged
# analyses return the complete paradigm with per/gen/num/mod filled by
# the suffix tables (e.g. 1st person carries gen 'u', dual 2nd 'u'),
# which fully-specified requests silently reject. One call per variant
# replaces the full spec cross-product and yields exactly the cells
# the resource knows.
VERB_VARIANTS = [
    {},
    {"prc1": "bi_prep"},
    {"prc1": "li_prep"},
    {"prc1": "sa_fut"},
    {"prc2": "wa_conj"},
    {"prc2": "fa_conj"},
    {"prc3": ">a_ques"},
]
NOUN_VARIANTS = [
    {},
    {"prc0": "Al_det"},
    {"prc1": "bi_prep"},
    {"prc0": "Al_det", "prc2": "wa_conj"},
    {"enc0": "1s_poss"},
    {"enc0": "3ms_poss"},
    {"enc0": "1p_poss"},
]
VERB_KEYS = ("pos", "asp", "per", "gen", "num", "vox", "mod", "prc0", "prc1", "prc2", "prc3", "enc0")
NOUN_KEYS = ("pos", "gen", "num", "cas", "stt", "prc0", "prc1", "prc2", "prc3", "enc0")


def _worker(args: tuple[str, str]) -> list[tuple[dict, str]]:
    lex, pos = args
    gen = _get_generator()
    variants = VERB_VARIANTS if pos == "v" else NOUN_VARIANTS
    keys = VERB_KEYS if pos == "v" else NOUN_KEYS
    db_pos = {"v": "verb", "adj": "adj", "n": "noun"}[pos]
    rows = []
    for extra in variants:
        spec = {"pos": db_pos, **extra}
        try:
            analyses = gen.generate(lex, spec)
        except Exception:
            continue
        for a in analyses:
            form = a.get("diac")
            if not form:
                continue
            feat = {k: a[k] for k in keys if a.get(k) not in (None, "", "na", "0")}
            rows.append((feat, form))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yalla-dir", type=Path, default=Path("/tmp/yallamorph"))
    ap.add_argument("--out", type=Path, default=Path("data/yallamorph-aux/lines.txt"))
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    lemmas = load_lemmas(args.yalla_dir)
    print({k: len(v) for k, v in lemmas.items()})

    gen = make_generator()
    in_lex = {pos: {lex for lex in lm if lex in gen._db.lemma_hash} for pos, lm in lemmas.items()}
    print("yallamorph-sample in-lexicon:", {k: len(v) for k, v in in_lex.items()})

    TARGETS = {"v": 4000, "n": 6000, "adj": 3000}
    db_by_pos = {"verb": "v", "noun": "n", "adj": "adj"}
    inventory: dict[str, list[str]] = {p: [] for p in db_by_pos.values()}
    for lex, stems in gen._db.lemma_hash.items():
        p = db_by_pos.get(stems[0].get("pos", ""))
        if p:
            inventory[p].append(lex)

    selected: dict[str, set[str]] = {}
    for pos, target in TARGETS.items():
        chosen = set(in_lex[pos])
        rest = sorted(set(inventory[pos]) - chosen)
        chosen.update(random.Random(42).sample(rest, max(0, target - len(chosen))))
        selected[pos] = chosen
    print("selected lemmas:", {k: len(v) for k, v in selected.items()})

    gold = parse_fewshot_gold(args.yalla_dir)
    print(f"[validate] {len(gold)} gold few-shot examples")
    n_ok = n_tot = 0
    for g in gold:
        feat = g["feat"]
        spec = {
            "pos": Y2C["pos"].get(feat.get("POS", ""), ""),
            "gen": Y2C["gen"].get(feat.get("GENDER", ""), ""),
            "num": Y2C["num"].get(feat.get("NUMBER", ""), ""),
        }
        if feat.get("PERSON"):
            spec["per"] = Y2C["per"][feat["PERSON"]]
        if feat.get("ASPECT"):
            spec["asp"] = Y2C["asp"][feat["ASPECT"]]
        if feat.get("MOD"):
            spec["mod"] = Y2C["mod"][feat["MOD"]]
        if feat.get("VOICE"):
            spec["vox"] = Y2C["voz"][feat["VOICE"]]
        if feat.get("CASE"):
            spec["cas"] = Y2C["cas"][feat["CASE"]]
        if feat.get("STATE"):
            spec["stt"] = Y2C["stt"][feat["STATE"]]
        for pname in sorted(PROC_SLOTS):
            v = feat.get(pname, "0")
            if v != "0":
                key = {"Al_det": "prc0", "bi_prep": "prc1", "li_prep": "prc1",
                       "wa_conj": "prc2", "fa_conj": "prc2", ">a_ques": "prc3"}.get(v)
                if key:
                    spec[key] = v
        for pname in sorted(feat):
            if pname.startswith("enclitics_0") and feat[pname] != "0":
                spec["enc0"] = feat[pname]
        spec = {k: v for k, v in spec.items() if v}
        try:
            forms = {a.get("diac") for a in gen.generate(feat["LEMMA"], spec)}
        except Exception as e:
            print(f"[validate] ERROR {g['file']}: {spec} -> {e}")
            continue
        hit = bool(forms & g["forms"])
        n_ok += hit
        n_tot += 1
        if not hit:
            print(f"[miss] {g['file']}: {spec} -> ours={sorted(f for f in forms if f)[:3]} gold={sorted(g['forms'])}")
    print(f"[validate] exact-form match {n_ok}/{n_tot}")
    if args.validate_only:
        return
    if n_tot == 0 or n_ok / n_tot < 0.5:
        print("[abort] validation below 50% — fix feature mapping first")
        sys.exit(1)

    jobs = [(lex, pos) for pos, lm in selected.items() for lex in sorted(lm)]
    print(f"[gen] {len(jobs)} lemmas (underspecified paradigm generation)")

    rows: list[tuple[dict, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, part in enumerate(ex.map(_worker, jobs, chunksize=8)):
            rows.extend(part)
            if (i + 1) % 500 == 0:
                print(f"[gen] {i + 1}/{len(jobs)} lemmas -> {len(rows)} pairs", flush=True)
    print(f"[gen] {len(rows)} raw (feat, form) pairs")

    verb_rows = [r for r in rows if r[0].get("pos") == "verb"]
    nom_rows = [r for r in rows if r[0].get("pos") != "verb"]
    verb_cap = int(args.cap * 0.6)
    nom_cap = args.cap - verb_cap
    lines = build_lines(verb_rows, cap=verb_cap) + build_lines(nom_rows, cap=nom_cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pos_counts = Counter()
    for spec, _ in rows:
        pos_counts[spec.get("pos", "?")] += 1
    print(f"[out] {args.out} lines={len(lines)} raw-by-pos={dict(pos_counts)}")


if __name__ == "__main__":
    main()
