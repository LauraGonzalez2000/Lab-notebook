# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import os, sys
import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from pathlib import Path
import matplotlib as plt
pt.set_style('manuscript')

#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','Processed', "2022_12_14", '13-27-41')
NIdaq = np.load(os.path.join(datafolder, 'NIdaq.npy'), allow_pickle=True)
NIdaq_start = np.load(os.path.join(datafolder, 'NIdaq.start.npy'), allow_pickle=True)
face_cam_summary = np.load(os.path.join(datafolder, 'FaceCamera-summary.npy'), allow_pickle=True)
face_cam_ = np.load(os.path.join(datafolder, 'facemotion.npy'), allow_pickle=True)
visual_stim = np.load(os.path.join(datafolder, 'visual-stim.npy'), allow_pickle=True)
data = visual_stim[()]
#print(data)
data = NIdaq[()]   # extract the dictionary from the 0-d array
analog = data['analog']
plt.plot(data['analog'][0])#[12000:12500])

#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre','Processed', "2025_12_29", '16-23-48')
NIdaq = np.load(os.path.join(datafolder, 'NIdaq.npy'), allow_pickle=True)
print(NIdaq)
NIdaq_ = NIdaq[()]
print(NIdaq_.keys())
#data = NIdaq[()]   # extract the dictionary from the 0-d array
#analog = data['analog']
#plt.plot(data['analog'][0][12000:12500])

#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','2NatIm8contrasts', 'NDNF-Cre','Processed', "2026_04_21", '15-11-13')
NIdaq = np.load(os.path.join(datafolder, 'NIdaq.npy'), allow_pickle=True)
data = NIdaq[()]   # extract the dictionary from the 0-d array
analog = data['analog']
plt.plot(data['analog'][0][12000:12500])