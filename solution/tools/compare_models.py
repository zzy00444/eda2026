import numpy as np, pandas as pd
from experiment import DATA, BUILD, competition_score, pin_parts
f=pd.read_csv(DATA/'data'/'delay_estimate_ans.csv'); y=f.delay.to_numpy()
sx,sy,sp=pin_parts(f.From); tx,ty,tp=pin_parts(f.To); dist=np.abs(tx-sx)+np.abs(ty-sy)
o=np.random.default_rng(7).permutation(len(f)); v=o[300000:350000]; t=o[350000:400000]
def pred(name, idx):
    q=pd.read_csv(BUILD/(name+'.csv'))
    return q.set_index('row').loc[idx,'prediction'].to_numpy()
for idxname,idx in [('valid',v),('test',t)]:
    a=pred('arch_s3_900_45l511_'+idxname,idx); b=pred('arch_s3_900_1023_lr10_220_'+idxname,idx)
    print(idxname,'all',competition_score(y[idx],a),competition_score(y[idx],b))
    for lo,hi in [(0,10),(10,30),(30,100),(100,300),(300,1000)]:
        z=idx[(dist[idx]>=lo)&(dist[idx]<hi)]; aa=pred('arch_s3_900_45l511_'+idxname,z); bb=pred('arch_s3_900_1023_lr10_220_'+idxname,z)
        print(lo,hi,len(z),competition_score(y[z],aa),competition_score(y[z],bb), 'gain',competition_score(y[z],bb)-competition_score(y[z],aa))
