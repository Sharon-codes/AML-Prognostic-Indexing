"""Inspect downloaded public expression datasets before selecting test protocols."""
import io,gzip,urllib.request,csv,hashlib
import pandas as pd
urls={
'golub':'https://raw.githubusercontent.com/aamirmalik-dr/high-dimensional-genomics-ml/main/data/leukemia_sample.csv',
'geo':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE9nnn/GSE9476/matrix/GSE9476_series_matrix.txt.gz',
'golub_full':'https://hastie.su.domains/CASI_files/DATA/leukemia_big.csv'
}
for key,url in urls.items():
    try:
        b=urllib.request.urlopen(url,timeout=70).read()
        print('FILE',key,'bytes',len(b),'sha256',hashlib.sha256(b).hexdigest(),flush=True)
        if key=='golub':
            x=pd.read_csv(io.BytesIO(b))
            print('GOLUB',x.shape,'last_columns',x.columns[-5:].tolist(),'label_counts',x.iloc[:,-1].value_counts().to_dict(),flush=True)
        elif key=='geo':
            raw=gzip.decompress(b).decode('utf-8',errors='replace')
            lines=raw.splitlines()
            for l in lines[:190]:
                if l.startswith(('!Sample_title','!Sample_source_name','!Sample_characteristics','!Sample_geo_accession','!Sample_platform','!Sample_organism')):
                    vals=next(csv.reader([l],delimiter='\t'))
                    print('GEO META',vals[0], 'count',len(vals)-1,'unique_first',sorted(set(vals[1:]))[:16],flush=True)
            print('MATRIX START',next((l[:500] for l in lines if l.startswith('"ID_REF"')),None),'total_lines',len(lines),flush=True)
        else:print('BIG bytes okay',flush=True)
    except Exception as e:
        print('FAILED',key,repr(e),flush=True)
