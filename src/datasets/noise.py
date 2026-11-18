"""Key component written by the team (not a library call): SNR-exact white / pink noise mixing.

SNR_dB = 10*log10(P_signal / P_noise), so the noise is scaled by sqrt(P_s / (P_n * 10^(SNR/10))).
Works on a single waveform [T] or a batch [B, T] with one SNR per row (vectorised for GPU training).
"""
import torch


def colored_noise(shape, color="white", generator=None, device=None):
    n = torch.randn(shape, generator=generator, device=device)
    if color == "pink":  # 1/f power -> divide the amplitude spectrum by sqrt(f)
        spec = torch.fft.rfft(n)
        f = torch.arange(spec.shape[-1], device=n.device).clamp_min(1).float()
        n = torch.fft.irfft(spec / f.sqrt(), n=shape[-1])
    return n


def mix_noise(x, snr_db, color="white", generator=None):
    noise = colored_noise(x.shape, color, generator, None if generator is not None else x.device).to(x.device)
    snr = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device)
    if snr.dim():
        snr = snr.view(-1, *[1] * (x.dim() - 1))
    ps = x.pow(2).mean(-1, keepdim=True)
    pn = noise.pow(2).mean(-1, keepdim=True).clamp_min(1e-12)
    scale = torch.sqrt(ps / (pn * 10 ** (snr / 10)))
    return torch.where(ps > 1e-12, x + scale * noise, x)  # digital silence stays silent


def measured_snr(clean, noisy):
    return float(10 * torch.log10(clean.pow(2).mean() / (noisy - clean).pow(2).mean()))


def self_test():
    x = torch.sin(torch.linspace(0, 2000, 110250))
    for color in ["white", "pink"]:
        for snr in [10, 0, -5]:
            got = measured_snr(x, mix_noise(x, snr, color, torch.Generator().manual_seed(0)))
            assert abs(got - snr) < 1e-3, (color, snr, got)
    batch = mix_noise(x.repeat(4, 1), torch.tensor([10.0, 5, 0, -5]))
    assert [round(measured_snr(x, b)) for b in batch] == [10, 5, 0, -5]
    print("noise self-test passed (white/pink, scalar/batched SNR exact)")
