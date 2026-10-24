#!/usr/bin/env python3
"""Recompute stability and classification evidence and run a 10,000-replicate null.
Input is a recorded evidence snapshot, not patient expression or prediction arrays.
Usage: python jbcb_revision/reanalysis.py --input jbcb_revision/input_snapshot.json
Requires NumPy.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

def nogueira(histogram,runs,candidates):
    counts=[(int(k),int(v)) for k,v in histogram.items()]
    mean_k=sum(k*v for k,v in counts)/runs
    pi=mean_k/candidates
    if not 0<pi<1:raise ValueError("Invalid subset size or universe")
    variance=runs/(runs-1)/candidates*sum(v*(k/runs)*(1-k/runs) for k,v in counts)
    return 1-variance/(pi*(1-pi))

def wilson(s,n):
    if not n or not 0<=s<=n:raise ValueError("Invalid binomial counts")
    z=1.959963984540054
    p=s/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,mid-half),min(1,mid+half)]

def evaluate(e):
    m=int(e["runs"]);k=int(e["genes_per_run"]);universe=int(e["candidates"])
    h=e["histogram"]
    if sum(int(c)*v for c,v in h.items())!=m*k:raise ValueError("Run counts inconsistent")
    obs=nogueira(h,m,universe)
    metrics={}
    for cohort,c in e["cohorts"].items():
        tp,fn,tn,fp=(c[key] for key in ("tp","fn","tn","fp"))
        metrics[cohort]=dict(precision=tp/(tp+fp),sensitivity=tp/(tp+fn),
            specificity=tn/(tn+fp),precision_ci=wilson(tp,tp+fp),
            sensitivity_ci=wilson(tp,tp+fn),specificity_ci=wilson(tn,tn+fp),
            tp=tp,fn=fn,tn=tn,fp=fp)
    rng=np.random.default_rng(20261024)
    reps=10000
    sim=np.empty(reps);unique=np.empty(reps,dtype=int)
    for r in range(reps):
        freq=np.zeros(universe,dtype=np.int16)
        for _ in range(m):
            freq[rng.choice(universe,k,replace=False)]+=1
        binc=np.bincount(freq,minlength=m+1)
        sim[r]=nogueira({str(i):int(binc[i]) for i in range(1,len(binc)) if binc[i]},m,universe)
        unique[r]=(freq>0).sum()
    pairs=m*(m-1)//2
    shared=sum(v*int(key)*(int(key)-1)//2 for key,v in h.items())
    return dict(source_commit=e["source_commit"],
        method="New computations from archived feature-selection and classification counts",
        runs=m,genes_per_run=k,distinct_genes=sum(h.values()),
        nogueira_1000=obs,nogueira_16508_sensitivity=nogueira(h,m,16508),
        shared_gene_events=shared,run_pairs=pairs,
        mean_intersection=shared/pairs,pooled_jaccard=shared/(pairs*2*k-shared),
        random_null_replicates=reps,random_null_median=float(np.median(sim)),
        random_null_95=[float(v) for v in np.quantile(sim,[.025,.975])],
        random_null_p_ge_observed=float((1+sum(sim>=obs))/(reps+1)),
        random_null_unique_gene_median=float(np.median(unique)),
        random_null_unique_95=[float(v) for v in np.quantile(unique,[.025,.975])],
        no_gene_at_80_percent=max(map(int,h)) < .8*m,
        classification=metrics,
        evidence_boundary="This is not a newly fitted clinical classifier or within-platform validation.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path(__file__).with_name("input_snapshot.json"))
    p.add_argument("--out",type=Path,default=Path(__file__).with_name("recomputed.json"))
    a=p.parse_args()
    x=evaluate(json.loads(a.input.read_text(encoding="utf-8")))
    a.out.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(x,indent=2))
