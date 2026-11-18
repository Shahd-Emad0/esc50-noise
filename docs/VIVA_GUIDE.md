# Viva guide — every member should be able to answer these

**Why group split on `src_file`?** Several ESC-50 clips are cut from one Freesound recording; a random split puts near-duplicates in train and test and inflates scores (leakage).
**Why macro-F1?** It weights every class equally and exposes classes the model ignores; accuracy hides per-class failure.
**How is SNR set exactly?** gain = sqrt(Ps / (Pn·10^(SNR/10))); self-test checks measured SNR within 0.001 dB.
**Why pink noise only at test?** To check whether robustness generalises to a noise colour never seen (H4).
**How is the test set protected?** Selection uses validation only; `evaluate.py` runs once at the end; best/demo model chosen by validation.
**What does attention pooling do?** Softmax weights over 54 time frames per head → weighted sum; can focus on frames where the event exceeds the noise. Ablation replaces it with mean pooling.
**Why freeze AST?** 86 M parameters vs 1,280 clips → over-fitting and cost; only 39,986 head parameters are trained.
**Pretrained vs scratch?** Same EfficientNet-B0, same budget, only the initial weights differ.
**What does a seed change?** Initialisation, batch order, augmentation noise, dropout. The test noise is fixed.
**95 % CI with 3 seeds?** mean ± t(0.975, 2)=4.30 · SD/√3 on paired per-seed differences.
**Main failure modes?** Noise masking of short/quiet events; noisy clips labelled as noise-like classes (wind, rain, engine); intrinsic confusions; pink-noise shift.
