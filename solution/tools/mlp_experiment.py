"""Probe a compact dense model for branch-free inference."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy import sparse
sys.path.insert(0,str(Path(__file__).parent))
from experiment import DATA,BUILD,features,competition_score
from architecture import predict

frame=pd.read_csv(DATA/'data/delay_estimate_ans.csv')
order=np.random.default_rng(7).permutation(len(frame)); train=np.r_[order[:300000],order[400000:]][:300000]; valid=order[300000:350000]; test=order[350000:400000]
x,_=features(frame); base=predict(frame); x['architecture_delay']=base
numeric=['sx','sy','tx','ty','dx','dy','ax','ay','manhattan','max_axis','gap_x','gap_y','same_row','same_column','source_bit','target_bit','first_dx','first_dy','remaining_x','remaining_y','remaining_ax','remaining_ay','dx_mod','dy_mod','ax_mod','ay_mod','remaining_x_mod','remaining_y_mod','remaining_ax_mod','remaining_ay_mod','sy_mod100','ty_mod100','architecture_delay']
num=x[numeric].to_numpy(dtype=np.float32); scaler=StandardScaler().fit(num[train]); num=scaler.transform(num).astype(np.float32)
ids=[x[c].cat.codes.to_numpy() for c in ('source_port','target_port')]
onehot=np.zeros((len(frame),992),dtype=np.float32)
onehot[np.arange(len(frame)),ids[0]]=1
onehot[np.arange(len(frame)),496+ids[1]]=1
X=np.hstack((num,onehot))
y=frame.delay.to_numpy(dtype=np.float32)
scale=np.maximum(base,200).astype(np.float32); z=(y-base)/scale
model=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=1e-4,batch_size=4096,learning_rate_init=0.002,max_iter=30,shuffle=True,random_state=7,verbose=True)
started=time.time();model.fit(X[train],z[train]);print('train_seconds',time.time()-started,flush=True)
for name,idx in [('valid',valid),('test',test)]:
 p=base[idx]+model.predict(X[idx])*scale[idx]
 print(name,competition_score(y[idx],p),float(np.mean(np.abs(np.rint(p)-y[idx]))),flush=True)
np.savez(BUILD/'mlp_probe.npz',coefs=np.array(model.coefs_,dtype=object),intercepts=np.array(model.intercepts_,dtype=object),mean=scaler.mean_,scale=scaler.scale_)
