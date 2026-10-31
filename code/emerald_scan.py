# For every shard and row group: count DMSO_TF rows (metadata columns only).
import sys, json, fsspec, pyarrow.parquet as pq, pyarrow.compute as pc
from concurrent.futures import ThreadPoolExecutor
base="https://huggingface.co/datasets/tahoebio/EmeraldBay/resolve/main/expression_data/train-%05d-of-00116.parquet"
fs=fsspec.filesystem("http",client_kwargs={"trust_env":True})
def one(i):
    for t in range(5):
        try:
            with fs.open(base%i,'rb',block_size=2**20) as f:
                pf=pq.ParquetFile(f); out=[]
                for g in range(pf.metadata.num_row_groups):
                    tb=pf.read_row_group(g,columns=['drugname_drugconc'])
                    n=pc.sum(pc.match_substring(tb['drugname_drugconc'],'DMSO_TF')).as_py() or 0
                    out.append((g,tb.num_rows,n))
            return i,out
        except Exception as e:
            print('retry',i,e,flush=True)
    return i,None
res={}
with ThreadPoolExecutor(12) as ex:
    for i,o in ex.map(one,range(116)):
        res[i]=o; print('shard',i,None if o is None else sum(x[2] for x in o), None if o is None else sum(1 for x in o if x[2]>0), None if o is None else len(o),flush=True)
json.dump(res,open(sys.argv[1],'w'))
