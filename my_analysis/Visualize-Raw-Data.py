# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import sys, pathlib, os
physion_path = os.path.join(os.path.expanduser('~'), 'Programming/In_Vivo/physion/src')
sys.path += [physion_path]
import physion

import numpy as np

import physion.utils.plot_tools as pt
pt.set_style('dark')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

# %%
# load a datafile

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch2', 'NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]


dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

index = 1
filename = SESSIONS['files'][index]

data = Data(filename,
            verbose=False)

data.build_dFoF(**dFoF_options, verbose=False)
data.build_rawFluo(verbose=False)

# %% [markdown]
# ## Showing Field of View

# %%
fig, AX = pt.figure(axes=(3,1), 
                    ax_scale=(1.4,3), wspace=0.15)

from physion.dataviz.imaging import show_CaImaging_FOV
#
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3, # non-linearity to normalize image
                   ax=AX[0])
show_CaImaging_FOV(data, key='max_proj', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3, # non-linearity to normalize image
                   ax=AX[1])
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3,
                   roiIndex=range(data.nROIs), 
                   ax=AX[2])

# save on desktop
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'FOV.png'))

# %% [markdown]
# # Show Raw Data

# %%

# default plot
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
settings = find_default_plot_settings(data)
pt.set_style('manuscript')
_ = plot_raw(data, settings=settings, tlim=[1200,1300])

# %% [markdown]
# ## Full view

# %%
settings = {'Locomotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': '#1f77b4'},
            'FaceMotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': 'purple'},
            'Pupil': {'fig_fraction': 2,
                      'subsampling': 1,
                      'color': '#d62728'},
             'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           'subquantity': 'dF/F',
                           'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                           'color': '#2ca02c'}
           }
fig, AX = \
    plot_raw(data, 
             tlim=[100, data.t_dFoF[-1]], 
             settings=settings)


# %%
settings = find_default_plot_settings(data)
settings = {'CaImagingRaster' : dict(fig_fraction=3, 
                                           subsampling=1, 
                                           roiIndices='all',
                                           normalization='per-line',
                                           subquantity='dF/F'), 
             'VisualStim' : dict(fig_fraction=.5, 
                                      color='black')}

pt.set_style('manuscript')

_ = plot_raw(data, settings=settings, tlim=[0,1300])
