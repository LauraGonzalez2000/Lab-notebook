# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import os, sys
import numpy as np

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

from physion.dataviz.imaging import find_roi_coords, show_CaImaging_FOV

import physion.utils.plot_tools as pt

sys.path += ['../']
from Visual_Properties_analysis.Responsiveness_dynamics import generate_Resp_ROI_dict

from pathlib import Path

#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

data_s = []
for index in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][index]
    data = Data(filename,verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.init_visual_stim() #initializes visual stim (7 protocols (experiments) per file)
    data_s.append(data)

# %%

def get_colors(Resp_ROI_dict_c_all, protocol):
    roi_colors = []
    
    for key in Resp_ROI_dict_c_all.keys():
        category = Resp_ROI_dict_c_all[key][protocol]
        print(key, category)
        if category== ['Positive']:
            roi_colors.append("red")
        elif category== ['NS']:
            roi_colors.append("lightgray")
        if category== ['Negative']:
            roi_colors.append("deepskyblue")
    return roi_colors

#%% OVERALL VIEW PER FILE AND PER VISUAL_STIMULUS
data_s_ = data_s

protocols = ["static-patch", 
             "drifting-gratings",
             "Natural-Images-4-repeats",
             "moving-dots",
             "random-dots",
             "looming-stim"]

protocols = ["static-patch","drifting-gratings", "Natural-Images-4-repeats"]

fig, ax = pt.figure(axes=(len(data_s_),len(protocols)),
                         ax_scale=(2, 5))

for data_i, data in enumerate(data_s_):

    device = data.nwbfile.acquisition['CaImaging-TimeSeries'].imaging_plane.device 
    descr = device.name
    print(descr)
    #pix2um = float(descr.split("micronsPerPixel': {'XAxis': '")[1].split("', 'YAxis'")[0])
    pix2um = float(descr.split("micronsPerPixel'= {'XAxis'= '")[1].split("', 'YAxis'")[0])
    data.roi_positions = np.zeros((data.nROIs, 2))

    for roi in range(data.nROIs):
        x, y, _, _ = find_roi_coords(data, roi)
        data.roi_positions[roi,:] = pix2um*x, pix2um*y

    Resp_ROI_dict_c_all = generate_Resp_ROI_dict([data], metric="category", state='all')


    for protocol_i, protocol in enumerate(protocols):
        roi_colors = get_colors(Resp_ROI_dict_c_all, protocol)

        show_CaImaging_FOV(data, key='meanImg', 
                        cmap=pt.get_linear_colormap('k', 'tab:green'),
                        ax=ax[protocol_i][data_i], #[row][column]
                        with_ROI_annotation=False,
                        NL=3,
                        roiIndex=range(data.nROIs), 
                        roi_colors=roi_colors)
                        #    roiIndex=range(13))

#%% CALCULATE DISTANCES
#%% start calculating distances
print(np.linalg.norm(data.roi_positions[0,:] - data.roi_positions[1,:]))
print(np.linalg.norm(data.roi_positions[0,:] - data.roi_positions[2,:]))
print(np.linalg.norm(data.roi_positions[0,:] - data.roi_positions[3,:]))
# print(np.linalg.norm(
#     data.roi_positions[13,:] - data.roi_positions[2,:]))
