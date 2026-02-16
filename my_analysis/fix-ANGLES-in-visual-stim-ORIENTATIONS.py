# %%

### IMPORT THE VALUES TO REPLACE FROM A GIVEN DATAFILE ###

import os, sys, pathlib
import numpy as np

# %%
old_angles = [0.9, 23.27142857, 45.64285714,\
               68.01428571, 90.38571429,\
                112.75714286, 135.12857143, 157.5]
new_angles = np.linspace(0, 157.5, len(old_angles))

### LOOP OVER ALL FILES AND REPLACE THE VALUES ###
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'Processed')
filenames = pathlib.Path(datafolder).glob('**/visual-stim.npy')

#%%
for i, f in enumerate(filenames):
    if i<2000:
        print(i, ') ', f)
        try:
            stim = np.load(f, allow_pickle=True).item()
            stim['angle'] = np.array(stim['angle'])
            print("angles before : ", np.unique(stim['angle']))

            if 'angle' in stim:
                if len(np.unique(stim['angle']))==3:
                    # print('0', np.sum(np.array(stim['angle'])==0.))
                    # print('90', np.sum(np.array(stim['angle'])==90.))
                    # print('157.5', np.sum(np.array(stim['angle'])==157.5))
                    stim['angle'][stim['angle']==157.5]=0.

                elif len(np.unique(stim['angle']))==8:
                    for o, n in zip(np.unique(stim['angle']), new_angles):
                        stim['angle'][stim['angle']==o]=n

            print("angles after : ",np.unique(stim['angle']))
            np.save(f, stim)
        except BaseException as be:
            print(' [!!] Pb with ', f)
