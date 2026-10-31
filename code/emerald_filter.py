# Read only the row groups that contain DMSO_TF rows (from scan.json), keep the DMSO_TF rows, write one parquet per row group.
import sys, os, json, fsspec, pyarrow.parquet as pq, pyarrow.compute as pc
from concurrent.futures import ThreadPoolExecutor
base="https://huggingface.co/datasets/tahoebio/EmeraldBay/resolve/main/expression_data/train-%05d-of-00116.parquet"
fs=fsspec.filesystem("http",client_kwargs={"trust_env":True})
scan=json.load(open(sys.argv[1])); od=sys.argv[2]; os.makedirs(od,exist_ok=True)
jobs=[(int(i),g,n) for i,o in scan.items() for g,_,n in o if n>0]
def one(j):
    i,g,n=j; fn=os.path.join(od,'s%03d_g%d.parquet'%(i,g))
    if os.path.exists(fn): return j,'skip'
    for t in range(6):
        try:
            with fs.open(base%i,'rb',block_size=2**22) as f:
                tb=pq.ParquetFile(f).read_row_group(g)
            tb=tb.filter(pc.match_substring(tb['drugname_drugconc'],'DMSO_TF'))
            assert tb.num_rows==n,(tb.num_rows,n)
            pq.write_table(tb,fn+'.part',compression='zstd'); os.rename(fn+'.part',fn)
            return j,'ok'
        except Exception as e:
            print('retry',j,repr(e)[:200],flush=True)
    return j,'FAILED'
with ThreadPoolExecutor(12) as ex:
    for j,s in ex.map(one,jobs): print(j,s,flush=True)
print('DONE',len(jobs))
