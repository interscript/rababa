# 06 — ELECTRA pretraining (Hebrew)

Same as Arabic 06. Replaces MLM with Replaced-Token-Detection for
~2x sample efficiency.

## Tasks

### 6.1 Wire ELECTRA into modal_app.py
- Add `pretrain_method: mlm | electra` config flag
- Dispatch in pretrain function

### 6.2 Train Hebrew ELECTRA pretrain
- Use existing `training/electra.py`
- 6 epochs, same compute budget as MLM

### 6.3 Fine-tune + benchmark
- Compare ELECTRA-pretrained encoder vs MLM-pretrained encoder
- Acceptance: ELECTRA achieves lower DER at same fine-tune budget

## Acceptance
- [ ] ELECTRA val loss < MLM val loss at same epoch
- [ ] ELECTRA-pretrained Hebrew DER ≤ MLM-pretrained DER

## Files
- `modal_app.py` (add `pretrain_method` dispatch)
- `configs/rababa_hebrew_pretrain.yaml` (add `pretrain_method: electra`)
