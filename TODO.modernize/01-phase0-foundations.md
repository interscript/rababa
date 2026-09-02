# Phase 0 — Foundations

## Priority: P0 — unblocks all training

## Tasks

### 0.1 Modal auth
```bash
pip install modal
modal token new
modal secret create interscript-hf HF_TOKEN=...   # if needed
```
Verify `modal run` works on a trivial stub before committing to GPU.

### 0.2 Modal app skeleton (rababa + secryst)
Both repos get the same `modal_app.py` shape:

```python
import modal

app = modal.App("rababa")

# Volumes
datasets_vol = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)
models_vol = modal.Volume.from_name("rababa-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5", "transformers==4.46", "peft==0.13", "onnx==1.17",
                 "onnxruntime==1.20", "datasets==3.2", "accelerate==1.1",
                 "wandb==0.18", "omegaconf==2.3", "hydra-core==1.3")
    .copy_local_dir("./src", "/opt/rababa/src")
    .copy_local_file("./pyproject.toml", "/opt/rababa/pyproject.toml")
    .workdir("/opt/rababa")
    .run_commands("pip install -e .")
)

@app.function(gpu="A10G", timeout=60 * 60, image=image, volumes={"/datasets": datasets_vol})
def fetch_data(task: str):
    """Download dataset into the shared volume. Idempotent (skips if SHA matches)."""
    from src.rababa.datasets import fetch_task_dataset
    fetch_task_dataset(task, "/datasets")

@app.function(gpu="A10G", timeout=6 * 60 * 60, image=image,
              volumes={"/datasets": datasets_vol, "/checkpoints": checkpoints_vol})
def train_student(task: str, epochs: int = 5, fp16: bool = True):
    """Distill teacher → student. Logs to W&B. Checkpoints every 500 steps."""
    from src.rababa.training.distill import main
    main(task=task, epochs=epochs, fp16=fp16,
         data_root="/datasets", ckpt_root="/checkpoints")

@app.function(gpu="A10G", timeout=30 * 60, image=image,
              volumes={"/checkpoints": checkpoints_vol, "/models": models_vol})
def export_onnx(task: str, version: str, variant: str = "q8"):
    """Quantize student checkpoint → ONNX. Verify shape. Publish to volume."""
    from src.rababa.export import export_task_model
    export_task_model(task, version, variant,
                       ckpt_root="/checkpoints", model_root="/models")

@app.function(gpu="A10G", timeout=30 * 60, image=image,
              volumes={"/models": models_vol, "/datasets": datasets_vol})
def evaluate(task: str, version: str):
    """Run DER / CER on held-out test split. Print + return metrics."""
    from src.rababa.evaluate import evaluate_task_model
    return evaluate_task_model(task, version, "/models", "/datasets")
```

`secryst/modal_app.py` mirrors the same shape.

### 0.3 Dataset fetch pipelines

`rababa/src/rababa/datasets.py`:
```python
def fetch_task_dataset(task: str, root: str):
    """Tashkeela++ for Arabic, Dicta/NC for Hebrew."""
    if task == "rababa_arabic":
        url = "https://huggingface.co/datasets/tashkeela/resolve/main/data/"
        ...  # download, dedupe, validate
    elif task == "rababa_hebrew":
        ...  # Dicta API or NC dump
    else:
        raise ValueError(task)
```
SHA256 of `datasets/<task>/<split>.jsonl` is recorded; subsequent runs
skip download if hash matches.

`secryst/src/secryst/datasets.py`:
```python
def fetch_task_dataset(task: str, root: str):
    if task == "secryst_thai_ipa":
        ...  # Wiktionary dump + parser + IPA validator
```

### 0.4 Shared base config

`rababa/configs/base.yaml`:
```yaml
optimizer: adamw
scheduler: cosine
warmup_ratio: 0.03
fp16: true
seed: 42
log_every: 50
save_every: 500
grad_clip: 1.0
mixed_precision: bf16

distillation:
  alpha: 0.5
  temperature: 4.0

finetune:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj]
```

Task configs (`rababa_arabic.yaml`, etc.) extend base.

### 0.5 Manifest versioning

`ml-models/npm/models/manifest.json` becomes:
```json
{
  "schema_version": 1,
  "models": {
    "rababa_arabic": {
      "status": "preview",
      "version": "0.0.0",
      "cdn_base": "https://cdn.jsdelivr.net/gh/interscript/rababa@rababa_arabic-v{version}/models/",
      "github_base": "https://github.com/interscript/rababa/releases/download/rababa_arabic-v{version}/"
    },
    ...
  }
}
```
Versioning policy: only `major.minor.patch`. Breaking changes = major.
A single source of truth — bumping requires updating both the GitHub
release tag AND the manifest.

## Acceptance

- `modal run rababa/modal_app.py::fetch_data --task rababa_arabic` succeeds.
- SHA256 of `datasets/rababa_arabic/train.jsonl` committed to `DATASET_HASH`.
- CI test (`tests/test_fetch_data.py`) passes locally in CPU mode.

## Open questions
1. Is HF token needed for Tashkeela++? Check dataset card.
2. Do we want W&B tracking in production? Costs $ / has privacy implications.
