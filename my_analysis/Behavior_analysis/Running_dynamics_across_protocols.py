# %% [markdown]
# # Running dynamics accross protocols

# to complete

# %% [markdown]
### Load packages and define constants:

import sys, os

import numpy as np
import pandas as pd

from pathlib import Path
import matplotlib.colors as mcolors

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.utils import plot_tools as pt
#%%
# functions
def generate_behav_corr_ROI_dict(data_s, protocols=[''], subprotocols=False):

    #initialize
    nROIS = sum(data.nROIs for data in data_s)
    protocols = protocols = [p for p in data_s[0].protocols if (p != 'grey-20min')]
    #Resp_ROI_dict = {f"ROI_{i}": dict.fromkeys(protocols, None) for i in range(nROIS)}
    behav_corr_ROI_dict = {f"ROI_{i}": {p: [] for p in protocols} for i in range(nROIS)}
   
    #fill
    nROI_id = 0
    for data in data_s:
        print("\n\n data : ", data, "\n\n")

        if protocols == ['']:
            protocols = [p for p in data.protocols if (p != 'grey-20min')]

        for p in protocols: 

            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF', 'running_speed'])
            
            varied_params = [k for k in ep.varied_parameters.keys() if k != 'repeat']
            param_values = []
            cond = ep.find_episode_cond()
            running_speed = ep.running_speed #ep x values
            dFoF = ep.dFoF #ep x roi x values

            if subprotocols==True: 

                if len(varied_params) > 0 : 
                    param_values = ep.varied_parameters[varied_params[0]]
                    for param in varied_params:
                        cond_ = [ep.find_episode_cond(key=param,value=param_v) for param_v in param_values]
                    cond = [cond_p_i for cond_p_i in cond_]
                else: 
                    cond = [cond]

            else: 
                cond=[cond]

            for cond_i in cond : 
                running_speed_i = running_speed[cond_i]

                for roi_n in range(data.nROIs):

                    #dFoF_i = dFoF[:][roi_n][:]
                    dFoF_i = dFoF[cond_i, roi_n, :]

                    r_trials = [
                        np.corrcoef(running_speed_i[i], dFoF_i[i])[0, 1]
                        for i in range(dFoF_i.shape[0])
                    ]

                    r_mean = np.mean(r_trials)

                    behav_corr_ROI_dict[f"ROI_{nROI_id + roi_n}"][p].append(r_mean)

        nROI_id += data.nROIs
    
    return behav_corr_ROI_dict

def nonlinear_cmap(cmap, vmin, vmax, exp=0.5, N=256):
    """
    Smooth nonlinear remapping of a colormap with:
    - asymmetric vmin/vmax
    - midpoint fixed at value = 0
    - power-law control of color saturation
    """
    i = np.linspace(0, 1, N)
    mid = (0 - vmin) / (vmax - vmin)  # where value=0 lies in the data range

    i_nl = np.empty_like(i)

    left = i <= mid
    right = i > mid

    # left side (vmin -> 0)
    i_nl[left] = (0.5 * (i[left] / mid) ** exp)

    # right side (0 -> vmax)
    i_nl[right] = (0.5 + 0.5 * ((i[right] - mid) / (1 - mid)) ** exp)

    colors = cmap(i_nl)
    return mcolors.ListedColormap(colors)

cmap_graywarm = mcolors.LinearSegmentedColormap.from_list("graywarm",
                                                          ["#3b4cc0",  # blue (negative)
                                                           "#bdbbbb",  # mid gray (zero)
                                                           "#b40426"],   # red (positive)
                                                          N=256)
#%% CORR COEFF for behavior for each cell
###########################################################################
###########################################################################
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

data_s = []
for i in range(len(SESSIONS['files'])):
    data = Data(SESSIONS['files'][i], verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)

#%%
#corr_behav_ROI_dict = generate_Resp_ROI_dict(data_s, metric="value", state="all", subprotocols=False)
corr_behav_ROI_dict = generate_behav_corr_ROI_dict(data_s=data_s, subprotocols=True)

#%%
vmin = 0
vmax = 1

# Convert to matrix
df = pd.DataFrame.from_dict(corr_behav_ROI_dict).T

expanded_cols = []

for col in df.columns:
    expanded = df[col].apply(pd.Series)
    expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
    expanded_cols.append(expanded)

df = pd.concat(expanded_cols, axis=1)
#mapping = {'Positive': vmax, 'Negative': vmin, 'NS': 0}
#df_numeric = df.replace(mapping)

#df = df.sample(n=70) #zoom
#df = df.sort_values(by="looming-stim", ascending=False)

# ROI response vs STIM 
fig, AX = pt.figure(figsize=(5,5), 
                    ax_scale=(2, 10)) 
        
cmap_graywarm_nl = nonlinear_cmap(cmap_graywarm, vmin=vmin, vmax=vmax, exp = 0.7, N=256)

AX.imshow(df.values, 
        aspect='auto', 
        cmap= cmap_graywarm_nl, 
        vmin = vmin,
        vmax = vmax, 
        interpolation='nearest')

pt.bar_legend(AX, 
            colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
            colormap = cmap_graywarm_nl, #colormap=pt.binary, #pt.plt.cm.plasma #pt.plt.cm.coolwarm
            bar_legend_args={'fontsize':1},
            label='Amplitude response post-pre',
            X=np.arange(vmin, vmax+0.5, 0.5),
            bounds=[vmin, vmax],
            ticks = None,
            ticks_labels=None,
            no_ticks=False,
            orientation='vertical')

pt.set_plot(AX, 
            spines = ['bottom', 'left'],
            yticks=[0, len(df.index)],
            ylabel='ROI',
            xticks=range(len(df.columns)), 
            xticks_labels=df.columns,
            xticks_rotation=90,
            fontsize=8)

for i in np.arange(0.5, df.shape[1]):
    AX.axvline(x=i, color='black', linewidth=0.5)