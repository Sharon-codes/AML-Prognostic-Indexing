"""Reproducible nested AML feature selection experiments on two public datasets.

Runs as a GitHub Actions job with internet access. Uses full Golub AML-vs-ALL
and GEO GSE9476 AML-vs-healthy, keeping them as different targets. Feature
selection and scaler fits occur within each outer training fold.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import io, gzip, csv, hashlib, json, urllib.request, math, time, warnings
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
URLS={
 'Golub-1999':'https://hastie.su.domains/CASI_files/DATA/leukemia_big.csv',
 'GSE9476':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE9nnn/GSE9476/matrix/GSE9476_series_matrix.txt.gz',
}
SEED=20261024
K=15
CANDIDATES=80
PARTICLES=10
ITER=10
REPEATS=3
FOLDS=5
STABILITY_WEIGHT=.15
METHODS=['ANOVA','L1','ElasticNet','RandomForest','BPSO','BQPSO','Stability-BQPSO','Random15']

def fetch(url, out):
    p=out/'downloads'/url.rsplit('/',1)[-1]
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(url,timeout=120).read())
    b=p.read_bytes()
    print('DOWNLOAD',p.name,len(b),hashlib.sha256(b).hexdigest(),flush=True)
    return b, {'download_url':url,'file':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def load_golub(b):
    df=pd.read_csv(io.BytesIO(b),low_memory=False)
    names=[c for c in df.columns if c.split('.')[0].upper() in ('AML','ALL')]
    assert len(names)==72,(df.shape,df.columns[:4].tolist(),names[:4],len(names))
    x=df[names].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64).T
    y=np.array([int(c.upper().startswith('AML')) for c in names],dtype=np.int8)
    x=np.where(np.isfinite(x),x,np.nan)
    features=np.array([f'golub_feature_{j:05d}' for j in range(x.shape[1])])
    labels=np.array(['AML' if v==1 else 'ALL' for v in y])
    metadata={'n_aml':int(y.sum()),'n_comparator':int((y==0).sum()),'labels':'AML vs ALL','technology':'Affymetrix Hgu6800','annotation':'feature indices (data republisher does not preserve probe-gene mapping)'}
    return x,y,features,labels,metadata

def load_geo(b):
    lines=gzip.decompress(b).decode('utf-8',errors='replace').splitlines()
    meta={}
    for l in lines:
        if l.startswith('!Sample_'):
            fields=next(csv.reader([l],delimiter='\t'))
            meta.setdefault(fields[0],[]).append(fields[1:])
    accession=meta['!Sample_geo_accession'][0]
    char=meta['!Sample_characteristics_ch1'][0]
    src=meta.get('!Sample_source_name_ch1',[['']*len(accession)])[0]
    assert len(accession)==64
    # This retrospective GEO study contrasts 26 diagnosed AML samples and 38 normal donors.
    y=np.array([int('aml' in s.lower() and 'diagnos' in s.lower()) for s in char],dtype=np.int8)
    if sum(y)!=26:
        raise ValueError(f'Unexpected metadata diagnosis labels: {Counter(char)}; positives={sum(y)}')
    start=next(i for i,line in enumerate(lines) if line.startswith('"ID_REF"'))
    end=next(i for i in range(start+1,len(lines)) if lines[i].startswith('!series_matrix_table_end'))
    df=pd.read_csv(io.StringIO('\n'.join(lines[start:end])),sep='\t')
    feats=df.iloc[:,0].astype(str).to_numpy()
    z=df.set_index(df.columns[0]).loc[:,accession].apply(pd.to_numeric,errors='coerce')
    x=z.to_numpy(dtype=np.float64).T
    labels=np.array(['AML' if v else 'healthy' for v in y])
    normal_group=np.array([s if not v else 'AML clinical blast' for s,v in zip(char,y)])
    metadata={'n_aml':int(y.sum()),'n_comparator':int((y==0).sum()),'labels':'AML vs healthy hematopoietic','technology':'Affymetrix GPL96','source_label_table':{'|'.join(k):int(v) for k,v in Counter(zip(labels,src)).items()},'normal_subtypes':{k:int(v) for k,v in Counter(normal_group[y==0]).items()}}
    return x,y,feats,normal_group,metadata

def choose_top(sc,k):
    vals=np.nan_to_num(np.asarray(sc),nan=-1e12,posinf=1e12,neginf=-1e12)
    return np.argsort(-vals,kind='stable')[:k]

def anova_scores(x,y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores,_=f_classif(x,y)
    return np.nan_to_num(scores,nan=0,posinf=1e6,neginf=0)

def make_classifier(seed):
    return LinearSVC(C=1.0,class_weight='balanced',dual=False,max_iter=3000,random_state=seed)

def fit_predict(xtr,ytr,xts,features,seed):
    scaler=StandardScaler()
    a=scaler.fit_transform(xtr[:,features]);b=scaler.transform(xts[:,features]);a=np.clip(a,-10,10);b=np.clip(b,-10,10)
    model=make_classifier(seed);model.fit(a,ytr)
    return model.decision_function(b),model.predict(b)

def anchor_scores(x,y,seed):
    rng=np.random.default_rng(seed);acc=np.zeros(x.shape[1],dtype=np.float64)
    for _ in range(6):
        ids=np.concatenate([rng.choice(np.flatnonzero(y==z),sum(y==z),replace=True) for z in (0,1)])
        scores=anova_scores(x[ids],y[ids]);acc[choose_top(scores,K)]+=1
    return acc/6

def inner_fitness(x,y,seed):
    cv=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=seed).split(x,y))
    cache={}
    def evaluate(selected):
        selected=tuple(sorted(map(int,selected)))
        if selected in cache:return cache[selected]
        scores=[]
        for it,iv in cv:
            z,pred=fit_predict(x[it],y[it],x[iv],list(selected),seed)
            scores.append(balanced_accuracy_score(y[iv],pred))
        value=float(np.mean(scores));cache[selected]=value
        return value
    return evaluate,cache

def optimize(x,y,seed,flavor,anchor=None):
    rng=np.random.default_rng(seed)
    dims=x.shape[1]
    evaluate,cache=inner_fitness(x,y,seed)
    P=PARTICLES;N=ITER;k=K
    pos=rng.normal(-1,1,size=(P,dims));vel=rng.normal(0,.15,size=(P,dims))
    personal=pos.copy();best_masks=[];best_scores=[]
    def to_mask(v):
        # stochastic binary selection repaired to a fixed 15-marker budget.
        noisy=v+rng.logistic(size=dims)
        return choose_top(noisy,k)
    def value(mask):
        base=evaluate(mask)
        if anchor is None:return base
        return (1-STABILITY_WEIGHT)*base+STABILITY_WEIGHT*float(np.mean(anchor[mask]))
    for i in range(P):
        m=to_mask(pos[i]);best_masks.append(m);best_scores.append(value(m))
    best_scores=np.asarray(best_scores);idx=int(np.argmax(best_scores));gbest=pos[idx].copy();gmask=best_masks[idx].copy();gb_score=best_scores[idx]
    for epoch in range(N):
        meanbest=np.mean(personal,axis=0)
        for j in range(P):
            if flavor=='BPSO':
                vel[j]=.7*vel[j]+1.5*rng.random(dims)*(personal[j]-pos[j])+1.5*rng.random(dims)*(gbest-pos[j])
                pos[j]=np.clip(pos[j]+vel[j],-8,8)
            else:
                phi=rng.random(dims);attractor=phi*personal[j]+(1-phi)*gbest
                alpha=.95-.45*(epoch/max(1,N-1))
                pos[j]=np.clip(attractor+rng.choice([-1,1],size=dims)*alpha*np.abs(meanbest-pos[j])*np.log(1/np.clip(rng.random(dims),1e-12,1)),-8,8)
            m=to_mask(pos[j]);score=value(m)
            if score>best_scores[j]+1e-12:
                best_scores[j]=score;personal[j]=pos[j].copy();best_masks[j]=m
            if score>gb_score+1e-12:
                gb_score=score;gbest=pos[j].copy();gmask=m.copy()
    return np.array(gmask,dtype=int),len(cache),float(gb_score)

def selectors(x,y,seed):
    # All feature screening and learned selection are inside the outer training portion.
    f=anova_scores(x,y)
    universe=choose_top(f,min(CANDIDATES,x.shape[1]))
    xt=x[:,universe]
    scaler=StandardScaler();xs=np.clip(scaler.fit_transform(xt),-10,10)
    ans={'ANOVA':universe[choose_top(f[universe],K)]}
    l1=LogisticRegression(penalty='l1',solver='liblinear',C=.2,max_iter=2000,class_weight='balanced',random_state=seed)
    l1.fit(xs,y);ans['L1']=universe[choose_top(np.abs(l1.coef_[0]),K)]
    en=LogisticRegression(penalty='elasticnet',solver='saga',C=.3,l1_ratio=.5,max_iter=2000,class_weight='balanced',random_state=seed)
    en.fit(xs,y);ans['ElasticNet']=universe[choose_top(np.abs(en.coef_[0]),K)]
    rf=RandomForestClassifier(n_estimators=150,min_samples_leaf=2,class_weight='balanced',n_jobs=1,random_state=seed)
    rf.fit(xs,y);ans['RandomForest']=universe[choose_top(rf.feature_importances_,K)]
    anchor=anchor_scores(xt,y,seed+301)
    evaluation_budgets={}
    for name,flavor,weight in [('BPSO','BPSO',None),('BQPSO','BQPSO',None),('Stability-BQPSO','BQPSO',anchor)]:
        mask,n_eval,obj=optimize(xs,y,seed+401,flavor,weight)
        ans[name]=universe[mask]
        evaluation_budgets[name]={'unique_inner_subsets':n_eval,'objective':obj}
    rng=np.random.default_rng(seed+1234)
    ans['Random15']=rng.choice(x.shape[1],K,replace=False)
    return ans,evaluation_budgets

def nogueira(sets,d):
    n=len(sets);k=np.mean([len(x) for x in sets]);pi=k/d
    c=Counter(j for s in sets for j in s)
    variance=n/(n-1)/d*sum((v/n)*(1-v/n) for v in c.values())
    return 1-variance/(pi*(1-pi))

def summary_run(data_name,x,y,feat,subtype,meta,out):
    nanmask=np.any(~np.isfinite(x),axis=0)
    # Do not impute from full data: discard non-finite probe columns, a label-blind rule.
    x=x[:,~nanmask];feat=feat[~nanmask]
    splits=list(RepeatedStratifiedKFold(n_splits=FOLDS,n_repeats=REPEATS,random_state=SEED).split(x,y))
    records=[];predictions=[];panel_ids={m:[] for m in METHODS};budgets=[]
    for i,(train,test) in enumerate(splits):
        t=time.time()
        chosen,bud=selectors(x[train],y[train],SEED+i*157)
        for name,ids in chosen.items():
            score,pred=fit_predict(x[train],y[train],x[test],ids,SEED+i)
            met=dict(dataset=data_name,method=name,repeat=i//FOLDS,fold=i%FOLDS,
                n_train=len(train),n_test=len(test),auc=float(roc_auc_score(y[test],score)),
                balanced_accuracy=float(balanced_accuracy_score(y[test],pred)),
                sensitivity=float(recall_score(y[test],pred,zero_division=0)),
                specificity=float(recall_score(1-y[test],1-pred,zero_division=0)),
                precision=float(precision_score(y[test],pred,zero_division=0)),f1=float(f1_score(y[test],pred,zero_division=0)))
            records.append(met)
            panel_ids[name].append(set(map(int,ids)))
            for j,sample_id in enumerate(test):
                predictions.append(dict(dataset=data_name,method=name,repeat=i//FOLDS,fold=i%FOLDS,
                    sample=int(sample_id),truth=int(y[sample_id]),decision=float(score[j]),predict=int(pred[j]),
                    comparator_group=str(subtype[sample_id])))
        budgets.append(dict(dataset=data_name,repeat=i//FOLDS,fold=i%FOLDS,**{m+'_n_fit':v['unique_inner_subsets'] for m,v in bud.items()}))
        print('SPLIT',data_name,i+1,'/',len(splits),'sec',round(time.time()-t,2),flush=True)
    df=pd.DataFrame(records);pf=pd.DataFrame(predictions)
    df.to_csv(out/f'{data_name}_folds.csv',index=False)
    pf.to_csv(out/f'{data_name}_predictions.csv',index=False)
    pd.DataFrame(budgets).to_csv(out/f'{data_name}_compute_budget.csv',index=False)
    row=[];freqrows=[];subrows=[]
    for method in METHODS:
        r=df.loc[df.method==method]
        st=nogueira(panel_ids[method],len(feat))
        jac=[len(a&b)/len(a|b) for a,b in combinations(panel_ids[method],2)]
        by_rep=[]
        for rep in range(REPEATS):
            part=pf[(pf.method==method)&(pf.repeat==rep)].sort_values('sample')
            assert len(part)==len(y) and len(set(part['sample']))==len(y)
            ytrue=part.truth.to_numpy();score=part.decision.to_numpy();ypred=part.predict.to_numpy()
            by_rep.append(dict(auc=roc_auc_score(ytrue,score),balanced_accuracy=balanced_accuracy_score(ytrue,ypred),
                sensitivity=recall_score(ytrue,ypred),specificity=recall_score(1-ytrue,1-ypred)))
        row.append(dict(dataset=data_name,method=method,n_patients=len(y),n_positive=int(sum(y)),features=len(feat),
            folds=len(splits),panel_size=K,auc_mean=float(np.mean([q['auc'] for q in by_rep])),auc_sd=float(np.std([q['auc'] for q in by_rep],ddof=1)),
            balanced_accuracy_mean=float(np.mean([q['balanced_accuracy'] for q in by_rep])),
            sensitivity_mean=float(np.mean([q['sensitivity'] for q in by_rep])),specificity_mean=float(np.mean([q['specificity'] for q in by_rep])),
            nogueira=float(st),mean_pairwise_jaccard=float(np.mean(jac))))
        frequency=Counter(j for a in panel_ids[method] for j in a)
        for feature_id,count in frequency.most_common(25):
            freqrows.append(dict(dataset=data_name,method=method,feature=str(feat[feature_id]),count=count,frequency=count/len(splits)))
        if data_name=='GSE9476':
            for group in sorted(set(subtype[y==0])):
                sub=pf[(pf.method==method)&(pf.truth==0)&(pf.comparator_group==group)]
                subrows.append(dict(method=method,control_type=group,n_profiles=int(sub['sample'].nunique()),n_test_predictions=len(sub),specificity_over_repeated_folds=float((sub.predict==0).mean())))
    pd.DataFrame(freqrows).to_csv(out/f'{data_name}_feature_frequency.csv',index=False)
    if subrows:pd.DataFrame(subrows).to_csv(out/f'{data_name}_normal_subtypes.csv',index=False)
    meta.update({'nonfinite_features_dropped':int(nanmask.sum()),'retained_features':int(x.shape[1]),'sample_count':int(len(y))})
    return row,meta

def main():
    out=Path('jbcb_revision/bench_results');out.mkdir(parents=True,exist_ok=True)
    allrow=[];prov=[];metas={}
    for dataset in URLS:
        raw,p=fetch(URLS[dataset],out);prov.append(dict(dataset=dataset,**p))
        if dataset=='Golub-1999':x,y,features,groups,meta=load_golub(raw)
        else:x,y,features,groups,meta=load_geo(raw)
        print('COHORT',dataset,'samples',x.shape,'positive',int(y.sum()),'metadata',meta,flush=True)
        rows,meta=summary_run(dataset,x,y,features,groups,meta,out)
        allrow.extend(rows);metas[dataset]=meta
        pd.DataFrame(allrow).to_csv(out/'benchmark_summary.csv',index=False)
        (out/'dataset_metadata.json').write_text(json.dumps(metas,indent=2))
    (out/'provenance.json').write_text(json.dumps(prov,indent=2))
    (out/'config.json').write_text(json.dumps({'seed':SEED,'folds':FOLDS,'repeats':REPEATS,'candidates':CANDIDATES,'panel_size':K,'particles':PARTICLES,'iterations':ITER,'stability_weight':STABILITY_WEIGHT,'methods':METHODS,'evaluation':'fully outer-held-out patient folds, within-dataset; no independent external test','source':'original study treated separately'},indent=2))
    print('BENCHMARK COMPLETE\n'+pd.DataFrame(allrow).to_string(index=False),flush=True)
if __name__=='__main__':main()