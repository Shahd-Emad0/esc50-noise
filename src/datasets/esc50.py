"""ESC-50: download, leakage audit, waveform cache, log-mel features and the fixed noisy evaluation sets."""
import hashlib
import os
import urllib.request
import zipfile
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

from .noise import mix_noise

ROOT = Path(__file__).resolve().parents[2]
QUICK = os.environ.get("ESC50_QUICK") == "1"          # smoke-test mode writes to experiments/_quick
EXP = ROOT / "experiments" / ("_quick" if QUICK else "")
DATA = ROOT / "data"
AUDIO = DATA / "ESC-50-master" / "audio"
MANIFEST = DATA / "split_manifest.csv"
MANIFEST_SHA = "eed5f196072c20295cf73b9061b5f2aed20facf890ca54b3f8ae1062bdcb8b2a"
CACHE = DATA / "cache" / "waveforms_22050.npy"
URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
SR, N = 22050, 5 * 22050
TRAIN_SNRS = [10, 5, 0, -5]
CONDITIONS = ["clean", "10dB", "5dB", "0dB", "-5dB", "pink_0dB"]
VAL_CONDITIONS = CONDITIONS[:5]  # pink noise is never seen in training or model selection


def download():
    if AUDIO.exists() and len(list(AUDIO.glob("*.wav"))) == 2000:
        return
    print("Downloading ESC-50 (~650 MB) from", URL)
    tmp = DATA / "esc50.zip"
    urllib.request.urlretrieve(URL, tmp)
    zipfile.ZipFile(tmp).extractall(DATA)
    tmp.unlink()
    assert len(list(AUDIO.glob("*.wav"))) == 2000


def load_manifest():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA, "split manifest changed"
    df = pd.read_csv(MANIFEST, dtype={"src_file": str})
    df["split"] = df.split.replace({"validation": "val"})
    assert df.filename.is_unique and len(df) == 2000
    assert df.groupby("src_file").split.nunique().max() == 1, "source-file leakage across splits"
    for s in ["train", "val", "test"]:
        assert df[df.split == s].target.nunique() == 50, f"missing class in {s}"
    return df


def class_names(df):
    return df.sort_values("target").drop_duplicates("target").category.tolist()


def prepare(x, src_sr, sr=SR):
    """Any audio array -> mono, resampled, first 5 s (zero-padded)."""
    x = np.asarray(x, dtype=np.float32)
    x = x.mean(1) if x.ndim == 2 else x
    if src_sr != sr:
        g = gcd(int(src_sr), sr)
        x = resample_poly(x, sr // g, int(src_sr) // g)
    n = 5 * sr
    return np.pad(x[:n], (0, max(0, n - len(x)))).astype(np.float32)


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return prepare(x, sr)


@lru_cache(None)
def waveforms():
    """All 2000 clips in manifest order, float16 [2000, 110250]; decoded once and cached to disk."""
    df = load_manifest()
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("Decoding 2000 clips to", CACHE)
        np.save(CACHE, np.stack([load_audio(AUDIO / f) for f in df.filename]).astype(np.float16))
    return torch.from_numpy(np.load(CACHE)), df


def parse_condition(c):
    return ("pink" if c.startswith("pink") else "white"), int(c.split("_")[-1][:-2])


@lru_cache(None)
def condition_set(split, cond):
    """Deterministic corrupted copy of a split: identical noise samples for every model and seed."""
    X, df = waveforms()
    idx = np.where(df.split == split)[0]
    if cond == "clean":
        return X[idx], idx
    color, snr = parse_condition(cond)
    out = []
    for i, f in zip(idx, df.filename.iloc[idx]):
        seed = int.from_bytes(hashlib.sha256(f"2026:{f}:{cond}".encode()).digest()[:8], "little") % (2**63 - 1)
        out.append(mix_noise(X[i].float(), snr, color, torch.Generator().manual_seed(seed)))
    return torch.stack(out).half(), idx


class LogMel(torch.nn.Module):
    """128-band log-mel (n_fft 2048, hop 512) with per-clip standardisation -> [B, 1, 128, 216]."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=2048, hop_length=512, n_mels=128)

    def forward(self, x):
        z = 10 * torch.log10(self.mel(x).clamp_min(1e-10))
        z = (z - z.mean((1, 2), keepdim=True)) / z.std((1, 2), keepdim=True).clamp_min(1e-6)
        return z.unsqueeze(1)
