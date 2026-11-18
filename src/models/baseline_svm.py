"""Baseline (Rania Hamdy Gomaa): 40 MFCC mean+std (80-d) -> StandardScaler -> RBF SVM (C=10)."""
import librosa
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def mfcc_features(waves):
    out = []
    for w in waves:
        m = librosa.feature.mfcc(y=np.asarray(w, dtype=np.float32), sr=22050, n_mfcc=40, n_fft=2048, hop_length=512)
        out.append(np.concatenate([m.mean(1), m.std(1)]))
    return np.stack(out)


def make_svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
