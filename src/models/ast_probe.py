"""Modern transformer (Roqaya Ebrahim Elghandour): frozen AudioSet AST (Gong et al. 2021) + trained LayerNorm/linear head.
Inputs arrive as the shared 22.05 kHz waveforms (same noise as every other model) and are resampled to 16 kHz."""
import torch
import torchaudio

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
REVISION = "f826b80d28226b62986cc218e5cec390b1096902"


class ASTProbe:
    def __init__(self, device):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.device = device
        self.extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION)
        self.net = ASTForAudioClassification.from_pretrained(
            MODEL_ID, revision=REVISION, num_labels=50, ignore_mismatched_sizes=True, use_safetensors=True).to(device).eval()
        self.resample = torchaudio.transforms.Resample(22050, 16000)

    def load_head(self, path, labels):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["labels"] == labels, "label mapping mismatch"
        self.net.classifier.load_state_dict(ck["classifier"])
        return self

    @torch.no_grad()
    def embed(self, wave):  # [B, T@22.05k] -> [B, 768]
        x = self.resample(wave.float())
        inp = self.extractor([w.numpy() for w in x], sampling_rate=16000, return_tensors="pt")["input_values"]
        return self.net.audio_spectrogram_transformer(inp.to(self.device)).pooler_output.cpu()

    @torch.no_grad()
    def head(self, z):
        return self.net.classifier(z.to(self.device)).cpu()

    def __call__(self, wave):
        return self.head(self.embed(wave))
