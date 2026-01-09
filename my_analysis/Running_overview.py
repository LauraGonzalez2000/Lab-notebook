# %% [markdown]
# # Running overview 

# %% [markdown]
### Load packages and define constants:

#%%
import sys, os
import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.behavior import population_analysis
import matplotlib.pyplot as plt

running_speed_threshold = 0.5  #cm/s


#%% Load Data
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_new')
SESSIONS = scan_folder_for_NWBfiles(datafolder)

#%% [markdown]
## Plot traces
#%%
fig, ax = plt.subplots(nrows=len(SESSIONS['files']), ncols=1, figsize=(23, 30), facecolor='white',
                       layout='constrained')

for i in range(len(SESSIONS['files'])):

    data = Data(SESSIONS['files'][i], verbose=False)
    speed = data.nwbfile.acquisition['Running-Speed'].data[:]
    ax[i].plot(speed, c='k')
    ax[i].axhline(running_speed_threshold, c='r')
    ax[i].set_xlabel('Time (ms)', c='k')
    ax[i].set_ylabel('Running speed (cm/s)', c='k')
    ax[i].set_facecolor('white')

#%% histogram 
#for i in range(len(SESSIONS['files'])):

data = Data(SESSIONS['files'][13], verbose=False)
speed = data.nwbfile.acquisition['Running-Speed'].data[:]
plt.hist(speed,bins=10, range = [0,1])


#%%
speeds = []

for f in SESSIONS['files']:
    data = Data(f, verbose=False)
    speeds.append(data.nwbfile.acquisition['Running-Speed'].data[:])

speeds = np.concatenate(speeds)

plt.hist(speeds, bins=30, range = [0,1])
plt.xlabel("Running speed")
plt.ylabel("Count")
plt.title("Running-speed histogram (all sessions pooled)")
plt.show()


#%% [markdown]
## Minimum time running 10% cutoff
#%%
#Fraction running

run_path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_run')
#run_path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_run')
if not os.path.exists(run_path):
    os.makedirs(run_path)

fracs_running = []
running_speed_threshold = 0.5

for f, filename in enumerate(SESSIONS['files']):
        data = Data(filename, verbose=False)

        if (data.nwbfile is not None) and ('Running-Speed' in data.nwbfile.acquisition):
            speed = data.nwbfile.acquisition['Running-Speed'].data[:]
            frac = 100*np.sum(speed>running_speed_threshold)/len(speed)
            fracs_running.append(frac)
            if frac>=10:
                save_path = os.path.join(run_path, os.path.basename(filename))
                # Example: save NWB file copy
                with open(filename, 'rb') as src, open(save_path, 'wb') as dst:
                    dst.write(src.read())
                
            #print(f"file {filename} :\n fraction running {frac}")


x= np.arange(0,len(fracs_running),1)
y = fracs_running
threshold = 10
colors = ['r' if val > threshold else 'blue' for val in y]
plt.bar(x, y, color=colors)
plt.axhline(threshold, c='r')
plt.xlabel('Recording #')
plt.ylabel('Frac. running (%)')

#%% fraction running depending on animal
population_analysis(SESSIONS['files'])
