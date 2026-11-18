"""Live demo: upload or record a sound, optionally add white/pink noise at a chosen SNR, get the top-5 classes.
python demo/app.py   -> open http://127.0.0.1:7860   (all inference is local; no external AI API)"""
import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.esc50 import AUDIO, EXP, SR, LogMel, class_names, load_manifest, prepare  # noqa: E402
from src.datasets.noise import mix_noise                                                     # noqa: E402
from src.models import load_trained                                                          # noqa: E402
from src.train import all_configs, device, run_dir                                           # noqa: E402

DF = load_manifest()
LABELS, DEV = class_names(DF), device()
DEPLOY = json.loads((EXP / "results" / "deploy.json").read_text())
CFGS = {c["name"]: c for c in all_configs() if c["role"] == "main" and c["model"] != "svm" and c["name"] in DEPLOY["seeds"]}
_cache = {}


def get_model(name):
    if name not in _cache:
        cfg, seed = CFGS[name], DEPLOY["seeds"][name]
        if cfg["model"] == "ast":
            from src.models.ast_probe import ASTProbe
            _cache[name] = ASTProbe(DEV).load_head(ROOT / "checkpoints" / "ast_probe" / str(seed) / "best_head.pth", LABELS)
        else:
            m = load_trained(cfg, run_dir(name, seed) / "model.pt", DEV)
            _cache[name] = lambda x: m(x.to(DEV)).cpu()
    return _cache[name]


def classify(audio, display, noise, snr):
    if audio is None:
        raise gr.Error("Upload or record a sound first.")
    sr, x = audio
    x = x.astype(np.float32) / (np.iinfo(x.dtype).max if np.issubdtype(x.dtype, np.integer) else 1.0)
    wave = torch.from_numpy(prepare(x, sr))
    if noise != "none":
        wave = mix_noise(wave, float(snr), noise, torch.Generator().manual_seed(0))
    name = next(n for n, c in CFGS.items() if c["display"] == display)
    with torch.no_grad():
        p = get_model(name)(wave[None]).softmax(-1)[0].numpy()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.imshow(LogMel()(wave[None])[0, 0].numpy(), origin="lower", aspect="auto", cmap="magma")
    ax.set(title="Log-mel spectrogram seen by the model", xlabel="time frames", ylabel="mel bins")
    fig.tight_layout()
    return {LABELS[i]: float(p[i]) for i in np.argsort(-p)[:5]}, fig, (SR, wave.numpy())


test_files = DF[DF.split == "test"].groupby("category").head(1).filename.tolist()[:8]
names = [c["display"] for n, c in CFGS.items()]
best = CFGS.get(DEPLOY["best"], next(iter(CFGS.values())))["display"]
with gr.Blocks(title="ESC-50 under noise") as app:
    gr.Markdown("# Environmental sound classification under noise\nESC-50 · 50 classes · first 5 s of the clip are used. "
                "Scores are softmax outputs (not calibrated); sounds outside the 50 classes still get one of the 50 labels.")
    with gr.Row():
        with gr.Column():
            inp = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Sound")
            model = gr.Dropdown(names, value=best, label="Model")
            noise = gr.Radio(["none", "white", "pink"], value="none", label="Add noise (stress test)")
            snr = gr.Slider(-10, 20, value=0, step=1, label="SNR (dB)")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Top-5 prediction")
            spec = gr.Plot(label="Spectrogram")
            heard = gr.Audio(label="Audio the model received")
    gr.Examples([[str(AUDIO / f)] for f in test_files], inputs=[inp], label="Test-set examples (never used in training)")
    btn.click(classify, [inp, model, noise, snr], [out, spec, heard])

if __name__ == "__main__":
    app.launch()
