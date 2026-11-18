from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_DIR / "audio"
METADATA_PATH = PROJECT_DIR / "meta" / "esc50.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
FEATURES_DIR = PROJECT_DIR / "features"
RESULTS_DIR = PROJECT_DIR / "results"

SAMPLE_RATE = 22050
DURATION_SECONDS = 5
N_CLASSES = 50
RANDOM_SEED = 42

N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

SNR_LEVELS_DB = [10, 5, 0, -5]
METRICS = ["accuracy", "macro_f1"]