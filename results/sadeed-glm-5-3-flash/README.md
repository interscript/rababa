# GLM-5.3-Flash on SadeedDiac-25 (2026-08-31)

First measurement of glm-5.3-flash on this benchmark. Protocol: neutral
completion prompt, temperature 0, max_tokens 8192, **reasoning_effort=low**
(explicitly pinned — absent/unrecognized values default to MAX; the 5.2-era
`thinking:{type:disabled}` knob is rejected by 5.3 with HTTP 400 code 1210
and must not be combined with reasoning_effort). Reasoning content is
present at effort=low (tripwire disclosed); checkpoint
/tmp/sadeed_glm53_clean.jsonl, 1,200/1,200 non-empty.

| Protocol | DER (CE) | 
|---|---|
| raw (their default output) | 8.5911 |
| projected zero-skip | **8.8995** |

Context: GLM-5.2 zero-skip scored 2.6911 (our reproduction, thinking
disabled). The 5.3 flash tier regresses on this benchmark to below
dedicated Sadeed-1.5B (7.2915) — supporting the dedicated-vs-frontier
positioning: diacritization quality is not carried by general-frontier
scale at the flash tier. Effort=high/max remains unmeasured (protocol
prefers plain completion; a max-effort run would not be protocol-matched
to the other rows).
