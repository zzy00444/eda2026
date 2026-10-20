import numpy as np
import pandas as pd
from experiment import DATA, pin_parts
d=pd.read_csv(DATA/'data'/'delay_estimate_ans.csv')
sx,sy,sp=pin_parts(d.From); tx,ty,tp=pin_parts(d.To)
z=np.abs(tx-sx)+np.abs(ty-sy)
o=np.random.default_rng(7).permutation(len(d)); t=o[350000:400000]
print([(lo,hi,int(((z[t]>=lo)&(z[t]<hi)).sum())) for lo,hi in [(0,10),(10,30),(30,100),(100,300),(300,1000),(1000,9999)]])
