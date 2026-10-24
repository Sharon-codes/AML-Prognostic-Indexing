"""Posthoc held-out paired differences and exploratory GPL96 probe annotation.

The primary benchmarking predictions are frozen. Report one full repeat as the
patient-level paired comparison without counting 3 appearances of each patient
as independent. Bootstrap resamples patient IDs conditional on fold-fitted models.
Gene expression analyses are descriptive, not independent external validation.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import io, urllib.request, gzip, csv, json, hashlib, warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

BASE=Path('jbcb_revision/bench_results');BASE.mkdir(exist_ok=True,parents=True)
SRC='https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz'
GEO='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE9nnn/GSE9476/matrix/GSE9476_series_matrix.txt.gz'
ORIGINAL='logs_and_output/remediated_master_biomarkers.txt'


def paired(d,repeat=0,boots=4000,seed=123):
    t=pd.read_csv(BASE/f'{d}_predictions.csv')
    sub=t[(t.repeat==repeat)&t.method.isin(['BQPSO','Stability-BQPSO'])]
    left=sub[sub.method=='BQPSO'].sort_values('sample').reset_index(drop=True)
    right=sub[sub.method=='Stability-BQPSO'].sort_values('sample').reset_index(drop=True)
    assert left['sample'].equals(right['sample'])
    y=left.truth.to_numpy(); s0=left.decision.to_numpy();s1=right.decision.to_numpy();
    p0=left.predict.to_numpy();p1=right.predict.to_numpy()
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    dif_auc=[];dif_bal=[]
    for _ in range(boots):
        chosen=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        dif_auc.append(roc_auc_score(y[chosen],s1[chosen])-roc_auc_score(y[chosen],s0[chosen]))
        dif_bal.append(balanced_accuracy_score(y[chosen],p1[chosen])-balanced_accuracy_score(y[chosen],p0[chosen]))
    return dict(dataset=d,comparison='Stability-BQPSO minus BQPSO',chosen_repeat=repeat,
        n_unique_patients=len(y),bootstrap_replicates=boots,
        auc_delta=float(roc_auc_score(y,s1)-roc_auc_score(y,s0)),
        auc_delta_95_percentile=np.quantile(dif_auc,[.025,.975]).tolist(),
        balanced_accuracy_delta=float(balanced_accuracy_score(y,p1)-balanced_accuracy_score(y,p0)),
        balanced_accuracy_delta_95_percentile=np.quantile(dif_bal,[.025,.975]).tolist(),
        bootstrapping_scope='Patient resampling of one set of out-of-fold predictions; selected features and fold-fitted models held fixed. Conditional uncertainty only.')


def fetch_annotation():
    b=urllib.request.urlopen(SRC,timeout=120).read()
    raw=gzip.decompress(b).decode('utf-8',errors='replace')
    print('ANNOTATION',len(b),hashlib.sha256(b).hexdigest(),raw[:200],flush=True)
    lines=raw.splitlines()
    begin=next((i+1 for i,l in enumerate(lines) if l.startswith('!platform_table_begin')),None)
    if begin is None:
        begin=next((i for i,l in enumerate(lines) if l.startswith('ID\t') or l.startswith('"ID"')),None)
    if begin is None: raise RuntimeError('Missing GEO platform table')
    end=next((i for i in range(begin,len(lines)) if lines[i].startswith('!platform_table_end')),len(lines))
    annot=pd.read_csv(io.StringIO('\n'.join(lines[begin:end])),sep='\t',low_memory=False)
    print('ANNOT COLUMNS',annot.columns.tolist(),annot.shape,flush=True)
    probe_col=next(c for c in annot.columns if c.lower() in ['id','id_ref','probe set id'])
    gene_col=next((c for c in annot.columns if 'gene symbol' in c.lower() or c.lower().replace(' ','_')=='gene_symbol'),None)
    if gene_col is None: raise RuntimeError('No gene-symbol column in GPL96 annotation')
    annot['probe']=annot[probe_col].astype(str)
    annot['symbol']=annot[gene_col].astype(str).str.split('///',regex=False).str[0].str.strip()
    return annot[['probe','symbol']],{'annotation_url':SRC,'annotation_sha256':hashlib.sha256(b).hexdigest(),'probe_column':probe_col,'gene_column':gene_col}


def interpret():
    from sklearn.preprocessing import StandardScaler
    annot,provenance=fetch_annotation()
    m=dict(zip(annot.probe,annot.symbol))
    freq=pd.read_csv(BASE/'GSE9476_feature_frequency.csv')
    freq['gene_symbol']=freq.feature.map(m).fillna('')
    freq.to_csv(BASE/'GSE9476_probe_gene_annotations.csv',index=False)
    sg=freq[(freq.method=='Stability-BQPSO')].head(25)
    print('TOP SG genes',sg[['feature','gene_symbol','count']].to_string(index=False),flush=True)
    panel=Path(ORIGINAL).read_text().strip().splitlines()
    raw=urllib.request.urlopen(GEO,timeout=120).read()
    table=gzip.decompress(raw).decode('utf-8',errors='replace').splitlines()
    meta=[]
    for line in table:
        if line.startswith('!Sample_characteristics_ch1'):
            z=next(csv.reader([line],delimiter='\t'))
            meta.append(z[1:])
    labels=np.array([int('aml' in s.lower() and 'diagnos' in s.lower()) for s in meta[0]])
    start=next(i for i,line in enumerate(table) if line.startswith('"ID_REF"'))
    end=next(i for i in range(start+1,len(table)) if table[i].startswith('!series_matrix_table_end'))
    d=pd.read_csv(io.StringIO('\n'.join(table[start:end])),sep='\t')
    values=d.set_index(d.columns[0]).apply(pd.to_numeric,errors='coerce')
    valid=set(values.index.astype(str))
    by_gene={}
    for probe,symbol in m.items():
        symbol = str(symbol).strip()
        if probe not in valid or symbol.lower() in ('nan', 'none', '') or '///' in symbol:continue
        by_gene.setdefault(symbol,[]).append(probe)
    records=[]
    for gene in panel:
        probes=by_gene.get(gene,[])
        if not probes:
            records.append(dict(gene=gene,probe_count=0,mean_aml=np.nan,mean_control=np.nan,mean_difference=np.nan,welch_p=np.nan))
            continue
        mat=values.loc[probes].to_numpy(dtype=float)
        expr=np.nanmedian(mat,axis=0)
        pos,neg=expr[labels==1],expr[labels==0]
        p=float(ttest_ind(pos,neg,equal_var=False,nan_policy='omit').pvalue)
        records.append(dict(gene=gene,probe_count=len(probes),mean_aml=float(np.nanmean(pos)),mean_control=float(np.nanmean(neg)),mean_difference=float(np.nanmean(pos)-np.nanmean(neg)),welch_p=p))
    df=pd.DataFrame(records)
    sel=df.welch_p.notna()
    ps=df.loc[sel,'welch_p'].to_numpy()
    if len(ps):
        order=np.argsort(ps);sorted_p=ps[order];n=len(ps)
        adjusted=np.minimum.accumulate((sorted_p*n/(np.arange(n)+1))[::-1])[::-1]
        inverse=np.empty(n);inverse[order]=np.clip(adjusted,0,1)
        df.loc[sel,'bh_q_within_available_curated_genes']=inverse
    df.to_csv(BASE/'original_curated_panel_GSE9476_exploratory.csv',index=False)
    provenance.update({'geo_sha256':hashlib.sha256(raw).hexdigest(),'n_curated_mapped':int(sel.sum()),'n_curated_total':len(panel),'bh_family':'Curated-panel mapped genes with assay coverage','validation_boundary':'Unadjusted for subtype and cell composition; exploratory differential expression of pooled probes, not independent signature-performance validation.'})
    (BASE/'gene_annotation_provenance.json').write_text(json.dumps(provenance,indent=2))
    print('CURATED PANEL',df.to_string(index=False),flush=True)


def main():
    stats=[paired('Golub-1999'),paired('GSE9476')]
    (BASE/'paired_conditional_bootstrap.json').write_text(json.dumps(stats,indent=2))
    print('PAIRED AUDIT',json.dumps(stats,indent=2),flush=True)
    try:interpret()
    except Exception as e:
        (BASE/'gene_annotation_error.txt').write_text(repr(e)+'\n')
        print('ANNOTATION UNAVAILABLE',repr(e),flush=True)
if __name__=='__main__':main()