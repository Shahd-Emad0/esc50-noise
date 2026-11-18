"""Frozen AST feature probe; run from modern/. No test data accessed."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, str(Path.cwd() / 'src'))
from core import audit, load_audio, mix_noise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = Path('../results/split_manifest_v2.csv')
    df = audit(manifest)
    model_id = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    extractor = ASTFeatureExtractor.from_pretrained(model_id)
    net = ASTForAudioClassification.from_pretrained(
        model_id, num_labels=50, ignore_mismatched_sizes=True,
        use_safetensors=True).to(device)
    for param in net.parameters(): param.requires_grad = False
    net.eval()
    for param in net.classifier.parameters(): param.requires_grad = True
    out = Path('experiments/ast_probe') / str(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    signature = {'model': model_id, 'revision': getattr(net.config, '_commit_hash', None),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'audio_root': str(Path(a.audio).resolve()), 'seed': a.seed,
        'sr': 16000, 'duration': 5, 'noise': 'one copy per train clip, SNR in [10,5,0,-5]',
        'extractor': extractor.to_dict()}
    signature_json = json.dumps(signature, sort_keys=True, default=str)
    key = hashlib.sha256(signature_json.encode()).hexdigest()[:16]
    cache = out / ('features_' + key + '.pt')
    started = time.perf_counter()

    def encode(part, snr=None, training=False):
        rows = df[df.split == part].reset_index(drop=True)
        feats, targets = [], []
        with torch.inference_mode():
            for i, row in rows.iterrows():
                wave = load_audio(Path(a.audio) / row.filename, sr=16000)
                variants = [wave]
                if training:
                    level = [10,5,0,-5][int.from_bytes(hashlib.sha256(
                        f'{a.seed}:{row.filename}'.encode()).digest()[:4], 'little') % 4]
                    g = torch.Generator().manual_seed(a.seed * 10000 + i)
                    variants.append(mix_noise(wave, level, g))
                elif snr is not None:
                    # Same evaluation noise seed/filename key as EfficientNet;
                    # 16 kHz representation makes exact waveform noise samples differ.
                    seed = int.from_bytes(hashlib.sha256(
                        f'2026:{row.filename}:{snr}'.encode()).digest()[:8], 'little') % (2**63-1)
                    variants = [mix_noise(wave, snr, torch.Generator().manual_seed(seed))]
                for x in variants:
                    inputs = extractor(x.numpy(), sampling_rate=16000, return_tensors='pt')
                    values = inputs['input_values'].to(device)
                    z = net.audio_spectrogram_transformer(values).pooler_output
                    feats.append(z.cpu()); targets.append(int(row.target))
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print(f'Features {part} {snr}: {i+1}/{len(rows)}', flush=True)
        return torch.cat(feats), torch.tensor(targets)

    if cache.exists():
        data = torch.load(cache, map_location='cpu', weights_only=True)
        print('Loaded cached features', flush=True)
    else:
        data = {'train': encode('train', training=True)}
        for snr in [None,10,5,0,-5]:
            data['clean' if snr is None else f'{snr}dB'] = encode('val', snr=snr)
        torch.save(data, cache)
    # Clone outside inference mode so classifier autograd can save inputs.
    data = {k: (x.clone(), y.clone()) for k, (x,y) in data.items()}
    feature_seconds = time.perf_counter() - started
    (out/'protocol.json').write_text(json.dumps(signature, indent=2, default=str))
    opt = torch.optim.AdamW(net.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    gen = torch.Generator().manual_seed(a.seed)
    xtrain, ytrain = data['train']
    best = -1.; history = []; train_started = time.perf_counter()
    labels = df.sort_values('target').drop_duplicates('target').category.tolist()
    for epoch in range(1,a.epochs+1):
        net.classifier.train(); total = 0.
        order = torch.randperm(len(ytrain), generator=gen)
        for ids in order.split(64):
            x, y = xtrain[ids].to(device), ytrain[ids].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net.classifier(x), y)
            loss.backward(); opt.step(); total += loss.item() * len(y)
        net.classifier.eval(); metrics=[]
        with torch.no_grad():
            for condition in ['clean','10dB','5dB','0dB','-5dB']:
                x,y = data[condition]
                pred = torch.cat([net.classifier(b.to(device)).argmax(1).cpu() for b in x.split(64)])
                metrics.append({'condition':condition,'accuracy':accuracy_score(y,pred),
                    'macro_f1':f1_score(y,pred,labels=list(range(50)),average='macro',zero_division=0)})
        score = float(np.mean([m['macro_f1'] for m in metrics]))
        row = {'epoch':epoch,'loss':total/len(ytrain),'selection_macro_f1':score}
        row.update({m['condition']+'_accuracy':m['accuracy'] for m in metrics})
        history.append(row); print(row,flush=True)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if score > best:
            best=score
            torch.save({'classifier':net.classifier.state_dict(),'model_id':model_id,
                'model_revision':signature['revision'],'labels':labels,'seed':a.seed,
                'manifest_sha256':signature['manifest_sha256'],'epoch':epoch,
                'selection_macro_f1':score,'protocol':signature},out/'best_head.pth')
            pd.DataFrame(metrics).to_csv(out/'val_metrics.csv',index=False)
    (out/'cost.json').write_text(json.dumps({'feature_stage_wall_seconds':feature_seconds,
        'head_training_and_validation_seconds':time.perf_counter()-train_started,
        'hardware':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        'total_parameters':sum(p.numel() for p in net.parameters()),
        'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad)},indent=2))
    print('BEST VALIDATION RESULTS (not test):',flush=True)
    print(pd.read_csv(out/'val_metrics.csv').to_string(index=False))

if __name__ == '__main__':
    main()
