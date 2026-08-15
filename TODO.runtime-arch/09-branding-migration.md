# 09 — Naming/branding migration (proposal only — user approves each step)

Decided direction: Interscript is the single public brand; neural layer =
`interscript-ml`, "the phonological layer of Interscript". rababa =
vocalization lab credit; secryst = runtime lab credit (keeps onnxruntime
gem). The model format (IMF) is the adoptable artifact.

Steps (each a small PR, no repo renames without explicit approval):
1. READMEs across repos rewritten to the two-layer story with a diagram
2. @interscript/ml npm + interscript-ml pip published under that name
3. secryst gem README: "Ruby binding for interscript-ml"
4. Papers/RESULTS already use "phonological layer of Interscript" — align
   repo badges and citations to it
5. Optional (user call): rababa -> interscript/vocalization-lab alias

Acceptance: one coherent story from interscript.org to every repo README;
no broken links.
