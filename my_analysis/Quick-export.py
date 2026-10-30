# %%
import os
import numpy as np

# datafolder = "C:\Users\laura.gonzalez\Downloads\suite2p\suite2p\plane0"
datafolder = os.path.expanduser("~/Downloads/suite2p/suite2p/plane0")

stat = np.load(os.path.join(datafolder, 'stat.npy'), allow_pickle=True)
# %%
ops = np.load(os.path.join(datafolder, 'ops.npy'), allow_pickle=True)

# %%
F = np.load(os.path.join(datafolder, 'F.npy'), allow_pickle=True)
N = np.load(os.path.join(datafolder, 'Fneu.npy'), allow_pickle=True)
# %%
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter1d

fig, AX = plt.subplots(13, 1, figsize=(12,20))
for i in range(13):
    f = gaussian_filter1d(F[i], 2)
    f0 = np.max([f.min(), 1])
    dFoF = (f-f0)/f0
    AX[i].plot(dFoF)
    AX[i].axis('off')
    AX[i].plot([0,0], [0,1], 'k')
AX[0].plot([0,300], [0,0], 'k')
fig.savefig(os.path.expanduser("~/Desktop/fig.svg"))

# %%
