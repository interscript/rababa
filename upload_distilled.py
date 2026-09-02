"""Upload Hebrew distilled data to Modal volume."""
import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("data/hebrew-dictabert-distilled/train.txt", "/data/distilled.txt")
)

app = modal.App(name="rababa-upload", image=image)

@app.function(volumes={"/datasets": datasets_volume})
def upload():
    from pathlib import Path
    data = Path("/data/distilled.txt").read_text(encoding="utf-8")
    out = Path("/datasets/hebrew-dictabert-distilled/train.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data, encoding="utf-8")
    datasets_volume.commit()
    return {"lines": len(data.splitlines())}

@app.local_entrypoint()
def main():
    print(upload.remote())
