import hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import torch
import torchaudio
import timm
from torch.utils.data import Dataset


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def audit(path):
    df = pd.read_csv(path, dtype={'src_file': str})
    required = {'filename','target','category','src_file','split'}
    if not required <= set(df.columns): raise ValueError('Manifest columns missing')
    if df[list(required)].isna().any().any(): raise ValueError('Missing manifest values')
    if df.filename.duplicated().any(): raise ValueError('Duplicate filenames')
    if set(df.split) != {'train','val','test'} and set(df.split) != {'train','validation','test'}:
        raise ValueError('Expected train, val/validation, test')
    df['split'] = df.split.replace({'validation':'val'})
    if df.groupby('src_file').split.nunique().max() != 1: raise ValueError('Source leakage')
    if set(df.target) != set(range(50)): raise ValueError('Expected labels 0..49')
    if df.groupby('target').category.nunique().max()!=1 or df.groupby('category').target.nunique().max()!=1:
        raise ValueError('Inconsistent label mapping')
    for split in ['train','val','test']:
        if set(df[df.split==split].target)!=set(range(50)): raise ValueError('Missing class in '+split)
    return df


def load_audio(path, sr=22050):
    x, source_sr = sf.read(path, dtype='float32', always_2d=True)
    x=x.mean(axis=1)
    if source_sr!=sr:
        g=gcd(int(source_sr),sr); x=resample_poly(x,sr//g,int(source_sr)//g)
    n=5*sr
    return torch.tensor(np.pad(x[:n],(0,max(0,n-len(x)))),dtype=torch.float32)


def mix_noise(x, snr, generator=None):
    # Custom key component: exact power-ratio white noise mixing; no clipping.
    noise=torch.randn(x.shape, generator=generator, dtype=x.dtype)
    signal_power=x.square().mean()
    if signal_power <= 1e-12: return x.clone()
    scale=torch.sqrt(signal_power/(noise.square().mean().clamp_min(1e-12)*10**(snr/10)))
    return x+scale*noise


class Features:
    def __init__(self):
        self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=22050,n_fft=2048,
            hop_length=512,n_mels=128,power=2.0)
    def __call__(self,x):
        z=10*torch.log10(self.mel(x).clamp_min(1e-10))
        return ((z-z.mean())/z.std().clamp_min(1e-6)).unsqueeze(0)


class Sounds(Dataset):
    def __init__(self,df,audio,augment=False,snr=None,noise_seed=2026):
        self.df=df.reset_index(drop=True); self.audio=Path(audio)
        self.augment=augment; self.snr=snr; self.noise_seed=noise_seed; self.features=Features()
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]; x=load_audio(self.audio/row.filename)
        if self.snr is not None:
            key=f'{self.noise_seed}:{row.filename}:{self.snr}'.encode()
            seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1)
            x=mix_noise(x,self.snr,torch.Generator().manual_seed(seed))
        elif self.augment and torch.rand(()).item()<0.5:
            x=mix_noise(x,random.choice([10,5,0,-5]))
        return self.features(x),int(row.target),row.filename


def model(pretrained=False):
    return timm.create_model('efficientnet_b0',pretrained=pretrained,in_chans=1,num_classes=50)
