"""Final AST probe evaluation. Run from modern/. Never fits or selects a model."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()/'src'))
from core import audit, load_audio, mix_noise


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audio',required=True); a=ap.parse_args()
    manifest=Path('../results/split_manifest_v2.csv'); df=audit(manifest)
    fingerprint=hashlib.sha256(manifest.read_bytes()).hexdigest()
    labels=df.sort_values('target').drop_duplicates('target').category.tolist()
    seeds=[42,43,44]; saved={}
    for seed in seeds:
        path=Path(f'experiments/ast_probe/{seed}/best_head.pth')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['seed']!=seed or ck['manifest_sha256']!=fingerprint or ck['labels']!=labels:
            raise ValueError(f'Checkpoint protocol mismatch: {path}')
        saved[seed]=ck
    model_id=saved[42]['model_id']; revision=saved[42].get('model_revision')
    if any(c['model_id']!=model_id or c.get('model_revision')!=revision for c in saved.values()):
        raise ValueError('Backbone versions differ')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs={'revision':revision} if revision else {}
    extractor=ASTFeatureExtractor.from_pretrained(model_id,**kwargs)
    net=ASTForAudioClassification.from_pretrained(model_id,num_labels=50,
        ignore_mismatched_sizes=True,use_safetensors=True,**kwargs).to(device).eval()
    for param in net.parameters(): param.requires_grad=False
    for seed,ck in saved.items():
        expected=ck.get('protocol',{}).get('extractor')
        if expected:
            actual=extractor.to_dict()
            # Transformers may attach a different local serialization path.
            for key in ['feature_size','sampling_rate','num_mel_bins','max_length','do_normalize','mean','std','padding_value']:
                if key in expected and expected[key]!=actual.get(key):
                    raise ValueError(f'Extractor mismatch: {key}')
    out=Path('experiments/ast_probe/final_test'); out.mkdir(parents=True,exist_ok=True)
    test=df[df.split=='test'].reset_index(drop=True)
    missing=[f for f in test.filename if not (Path(a.audio)/f).is_file()]
    if missing: raise FileNotFoundError(missing[:5])
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    # Warm up full network; no test labels used.
    with torch.inference_mode():
        zero=extractor(np.zeros(80000,dtype=np.float32),sampling_rate=16000,return_tensors='pt')['input_values'].to(device)
        net(input_values=zero); sync()
    all_metrics=[]; costs=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for snr in [None,10,5,0,-5]:
        condition='clean' if snr is None else f'{snr}dB'
        zs=[]; backbone_seconds=0.; start_all=time.perf_counter()
        with torch.inference_mode():
            for i,row in test.iterrows():
                wave=load_audio(Path(a.audio)/row.filename,sr=16000)
                if snr is not None:
                    noise_seed=int.from_bytes(hashlib.sha256(f'2026:{row.filename}:{snr}'.encode()).digest()[:8],'little')%(2**63-1)
                    wave=mix_noise(wave,snr,torch.Generator().manual_seed(noise_seed))
                inputs=extractor(wave.numpy(),sampling_rate=16000,return_tensors='pt')['input_values'].to(device)
                sync(); started=time.perf_counter()
                z=net.audio_spectrogram_transformer(inputs).pooler_output
                sync(); backbone_seconds+=time.perf_counter()-started
                zs.append(z.cpu())
                if (i+1)%100==0: print(f'Test features {condition}: {i+1}/{len(test)}',flush=True)
        extraction_wall=time.perf_counter()-start_all
        z=torch.cat(zs); y=test.target.to_numpy()
        for seed in seeds:
            net.classifier.load_state_dict(saved[seed]['classifier']); net.classifier.eval()
            preds=[]; head_seconds=0.
            with torch.inference_mode():
                for b in z.split(1):
                    b=b.to(device); sync(); started=time.perf_counter()
                    logits=net.classifier(b); sync(); head_seconds+=time.perf_counter()-started
                    preds.append(int(logits.argmax(1).item()))
            pred=np.array(preds); directory=out/str(seed); directory.mkdir(exist_ok=True)
            prefix=directory/condition
            row={'seed':seed,'condition':condition,'n':len(y),'selected_epoch':saved[seed]['epoch'],
                'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,labels=list(range(50)),average='macro',zero_division=0),
                'model_only_ms_per_clip':1000*(backbone_seconds+head_seconds)/len(y)}
            all_metrics.append(row)
            frame=pd.DataFrame({'filename':test.filename,'true':y,'predicted':pred,
                'true_category':[labels[v] for v in y],'predicted_category':[labels[v] for v in pred]})
            frame.to_csv(str(prefix)+'_predictions.csv',index=False)
            errors=frame[frame.true!=frame.predicted]; errors.to_csv(str(prefix)+'_errors.csv',index=False)
            errors.groupby(['true_category','predicted_category']).size().rename('count').sort_values(ascending=False).to_csv(str(prefix)+'_error_pairs.csv')
            report=classification_report(y,pred,labels=list(range(50)),target_names=labels,output_dict=True,zero_division=0)
            pd.DataFrame(report).T.to_csv(str(prefix)+'_per_class.csv')
            cm=confusion_matrix(y,pred,labels=list(range(50)))
            pd.DataFrame(cm,index=labels,columns=labels).to_csv(str(prefix)+'_confusion.csv')
            fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(cm); fig.colorbar(im,ax=ax)
            ax.set(xlabel='Predicted class ID',ylabel='True class ID',title=f'AST seed {seed}: {condition}')
            fig.savefig(str(prefix)+'_confusion.png',dpi=150,bbox_inches='tight'); plt.close(fig)
            print(row,flush=True)
        costs.append({'condition':condition,'feature_extraction_wall_seconds':extraction_wall,
            'backbone_forward_seconds':backbone_seconds})
    result=pd.DataFrame(all_metrics); result.to_csv(out/'all_seed_metrics.csv',index=False)
    summary=result.groupby('condition').agg(seeds=('seed','nunique'),accuracy_mean=('accuracy','mean'),
        accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reindex(['clean','10dB','5dB','0dB','-5dB'])
    summary.to_csv(out/'test_summary.csv')
    # Full deployment state includes backbone, unlike the head-only checkpoints.
    num_params=sum(p.numel() for p in net.parameters())
    (out/'evaluation_protocol.json').write_text(json.dumps({'manifest_sha256':fingerprint,
        'model_id':model_id,'revision':revision,'seeds':seeds,'sr':16000,'seconds':5,'noise_seed':2026,
        'conditions':['clean','10dB','5dB','0dB','-5dB'],'hardware':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        'total_parameters':num_params,'full_state_tensor_bytes':sum(v.numel()*v.element_size() for v in net.state_dict().values()),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'timing_note':'Batch size 1, warmed model-only forward: backbone plus head. Excludes preprocessing, transfers, disk I/O. Backbone timing shared across identical frozen backbones; not three independent latency measurements.',
        'feature_stage_costs':costs,'comparison_note':'AST mixes at 16 kHz. EfficientNet used 22.05 kHz, so noise waveforms differ despite identical filename seeds/SNR. Treat prior cross-model comparison as exploratory.'},indent=2))
    print('\nFINAL TEST SUMMARY (mean and sample SD):\n'+summary.to_string(),flush=True)
    print('Saved to:',out.resolve())

if __name__=='__main__': main()
