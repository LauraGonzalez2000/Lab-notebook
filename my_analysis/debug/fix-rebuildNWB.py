# %%

### IMPORT THE VALUES TO REPLACE FROM A GIVEN DATAFILE ###

import os, sys, pathlib
import numpy as np
import shutil

# %%

keys_to_change = {
    "contrast": 1.0,
    "x-center": 0.0,
    "y-center": 0.0,
    "radius": 300.,
    "bg-color":0.5
}
PROTOCOL_ID = 4 # Natural-Images have protocol ID = 4

# %%
### Always restart from the same folder
original = os.path.expanduser('~/Desktop/NDNF-WT-Dec-2022-ORIGINAL/Processed')

new_folder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022', 'Processed')
if os.path.isdir(new_folder):
    shutil.rmtree(new_folder) # remove previous one

#%%
shutil.copytree(original, new_folder) #copy from original

#%%
### LOOP OVER ALL FILES AND REPLACE THE VALUES ###

filenames = pathlib.Path(new_folder).glob('**/visual-stim.npy')

for i, f in enumerate(filenames):

    print(i, ') ', f)

    stim = np.load(f, allow_pickle=True).item().copy()

    for v in np.flatnonzero(\
        stim['protocol_id']==PROTOCOL_ID):

        for key, new_val in keys_to_change.items():
            stim[key][v] = new_val

    np.save(f, stim)
