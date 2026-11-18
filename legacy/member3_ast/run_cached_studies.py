"""Supplementary validation-only head studies. Never reads final_test or changes final heads."""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/ast_probe'); a=ap.parse_args()
    root=Path(a.root); output=root/'supplementary_studies'; output.mkdir(exist_ok=True)
    all_rows=[]
    for seed in [42,43,44]:
        files=list((root/str(seed)).glob('features_*.pt'))
        if len(files)!=1: raise ValueError(f'Expected one cache for seed {seed}; found {len(files)}')
        data=torch.load(files[0],map_location='cpu',weights_only=True)
        x,y=data['train']; x=x.clone(); y=y.clone()
        if len(y)!=2560 or not torch.equal(y[::2],y[1::2]):
            raise ValueError('Expected clean/noisy interleaved original cache')
        for name,lr,aug in [('reference',1e-3,True),('lr_3e4',3e-4,True),('lr_3e3',3e-3,True),('no_noise',1e-3,False)]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            # Explicit initialization: all study arms share the same new head.
            # Original HF training consumed RNG while creating its full model,
            # so these reference runs are not claimed to reproduce old heads.
            head=torch.nn.Sequential(torch.nn.LayerNorm(x.shape[1]),torch.nn.Linear(x.shape[1],50))
            torch.nn.init.normal_(head[1].weight,std=.02); torch.nn.init.zeros_(head[1].bias)
            opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-3)
            gen=torch.Generator().manual_seed(seed)
            best=-1; best_rows=None; history=[]; final_epoch=None
            for epoch in range(1,31):
                head.train(); order=torch.randperm(len(y),generator=gen)
                # Equal number of examples and optimizer steps: replace each
                # noisy partner by its clean partner in the no-noise arm.
                ids=order if aug else (order//2)*2
                for b in ids.split(64):
                    opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(head(x[b]),y[b]);loss.backward();opt.step()
                rows=[]; head.eval()
                with torch.no_grad():
                    for condition in ['clean','10dB','5dB','0dB','-5dB']:
                        vx,vy=data[condition]; pred=head(vx.clone()).argmax(1)
                        rows.append({'study':name,'seed':seed,'condition':condition,
                            'accuracy':accuracy_score(vy,pred),
                            'macro_f1':f1_score(vy,pred,labels=list(range(50)),average='macro',zero_division=0)})
                score=float(np.mean([r['macro_f1'] for r in rows])); history.append({'epoch':epoch,'selection_macro_f1':score})
                if score>best:
                    best=score;best_rows=rows;final_epoch=epoch
            for r in best_rows:r['selected_epoch']=final_epoch
            all_rows.extend(best_rows)
            pd.DataFrame(history).to_csv(output/f'{name}_{seed}_history.csv',index=False)
            print(f'Seed {seed} {name}: best validation mean F1 {best:.4f}, epoch {final_epoch}',flush=True)
    frame=pd.DataFrame(all_rows);frame.to_csv(output/'all_metrics.csv',index=False)
    summary=frame.groupby(['study','condition']).agg(seeds=('seed','nunique'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'))
    summary.to_csv(output/'summary.csv')
    paired=frame.pivot(index=['seed','condition'],columns='study',values='macro_f1')
    delta=(paired.reference-paired.no_noise).rename('augmentation_gain_macro_f1')
    delta.to_csv(output/'paired_augmentation_deltas.csv')
    delta.groupby('condition').agg(['mean','std']).to_csv(output/'ablation_summary.csv')
    (output/'protocol.json').write_text(json.dumps({'status':'supplementary exploratory validation-only analysis after final test had been viewed',
        'final_model_changed':False,'seeds':[42,43,44],'epochs':30,'batch_size':64,'examples_per_epoch':2560,
        'optimizer_steps_per_epoch':40,'no_noise_arm':'Duplicate each clean embedding to match example/update budget',
        'encoder':'frozen original AudioSet AST','initialization':'same explicit random head initialization per seed across arms; not old head RNG initialization',
        'selection':'mean validation Macro F1 over five conditions','scratch_comparison':False},indent=2))
    print(summary.to_string())
    print('Saved supplementary validation-only studies:',output.resolve())

if __name__=='__main__':main()
