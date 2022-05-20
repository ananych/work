#%%
import pandas as pd
import numpy as np
import plotly.express as px

from scipy import signal




demand=pd.read_csv(r"C:\Users\reduc\Desktop\damand.csv")
data=[]
x=[]
group=[]

demand=np.array(demand)[1:,:]
for i in range(10):
    x.append(np.arange(i+1,i+13))
    data.append(demand[i,(i+1):(i+13)])
    group.append(np.repeat(i+1,12))
x=np.array(x).reshape(-1,)
data=np.array(data).reshape(-1,)
group=np.array(group).reshape(-1,)

all=np.concatenate([x,data,group])
all=all.reshape((120,-1),order='F')