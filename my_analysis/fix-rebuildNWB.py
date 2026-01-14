# %%

### IMPORT THE VALUES TO REPLACE FROM A GIVEN DATAFILE ###

import os, sys, pathlib
import numpy as np

if False:
    sys.path += ['./physion/src']
    from physion.analysis.read_NWB\
                            import scan_folder_for_NWBfiles, Data
    from physion.analysis.process_NWB import EpisodeData

    filename='/Users/yann/CURATED/Cibele/PV-cells_WT_Young_V1/NWBs/2025_10_10-16-52-46.nwb'
    data = Data(filename)
    ep = EpisodeData(data)

    old_angles = ep.varied_parameters['angle']
    new_angles = np.linspace(0, 157.5, len(old_angles))
    print(new_angles)

# %%

old = {
    "contrast": None,
    "x-center": None,
    "y-center": None,
    "radius": None,
}

new = {
    "contrast": np.float64(1.0),
    "x-center": np.float64(0.0),
    "y-center": np.float64(0.0),
    "radius": np.int32(300),
}

### LOOP OVER ALL FILES AND REPLACE THE VALUES ###
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022')
filenames = pathlib.Path(datafolder).glob('**/visual-stim.npy')

for i, f in enumerate(filenames):

    if i<2000:
        print(i, ') ', f)

        try:
            
            stim = np.load(f, allow_pickle=True).item()
            
            keys = ["contrast", "x-center", "y-center", "radius"]

            for key in keys: 
                print(key)
                stim[key] = np.array(stim[key])

                print("old : ", stim[key])

                for i, value in enumerate(stim[key]):
                    if value == old[key]:
                        stim[key][i] = new[key]

                
                print("new : ", stim[key])

            print("oo")
            np.save(f, stim)
        except BaseException as be:
            print(' [!!] Pb with ', f)
