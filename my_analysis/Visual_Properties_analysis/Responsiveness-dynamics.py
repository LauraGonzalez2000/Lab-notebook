# %% [markdown]
# # Responsiveness dynamics

#%%
import sys, os
import matplotlib.pyplot as plt

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

sys.path += ['../']
import General_summary.alluvial as alluvial
from collections import Counter

#%% 
####################################################################################################
#################################### RESPONSIVENESS DYNAMICS #######################################
######################################## ACROSS PROTOCOLS ##########################################
####################################################################################################

def generate_Resp_ROI_dict(data_s, protocols):

    #initialize
    nROIS = sum(data.nROIs for data in data_s)
    Resp_ROI_dict = {f"ROI_{i}": {"static-patch": None,
                              "drifting-gratings": None,
                              "Natural-Images-4-repeats": None} for i in range(nROIS)}
    
    #fill
    nROI_id = 0
    for data in data_s:
        for p in protocols: 

            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])

            for roi_n in range(data.nROIs):

                t0 = max([0, ep.time_duration[0]-1.5])
                stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                        interval_post=[t0, t0+1.5],                                   
                                        test='ttest', 
                                        sign='both')
                roi_summary_data = pre_post_statistics(ep,
                                                episode_cond = ep.find_episode_cond(),
                                                response_args = dict(roiIndex=roi_n),
                                                response_significance_threshold=0.05,
                                                stat_test_props=stat_test_props,
                                                repetition_keys=list(ep.varied_parameters.keys()))
                if bool(roi_summary_data['significant'])==False:
                    category = 'NS'
                else: 
                    if roi_summary_data['value']>0:
                        category = "Positive"
                    else: 
                        category = "Negative"

                Resp_ROI_dict[f"ROI_{nROI_id +roi_n}"][p] = category

        nROI_id += data.nROIs

    return Resp_ROI_dict

def generate_input_data(Resp_ROI_dict, prot1, prot2, categories=("Positive", "NS", "Negative")):
    input_data = {src: {dst + "_": 0 for dst in categories} for src in categories}
    for ROI, responses in Resp_ROI_dict.items():
        src = responses[prot1]
        dst = responses[prot2] + "_"
        input_data[src][dst] += 1
    return input_data

#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
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
protocols = ["static-patch", "drifting-gratings", "Natural-Images-4-repeats"]
Resp_ROI_dict = generate_Resp_ROI_dict(data_s, protocols)
print(Resp_ROI_dict)
#%%
#prot1 = "static-patch"
#prot2 = "drifting-gratings"
#--------------------------------
prot1="drifting-gratings"
prot2="Natural-Images-4-repeats"
#---------------------------------
#prot1="Natural-Images-4-repeats"
#prot2="static-patch"

input_data = generate_input_data(Resp_ROI_dict, prot1, prot2)

colors = ["red", "grey", "green"]
src_label_override=["Negative", 'NS', 'Positive']
dst_label_override=["Negative_", 'NS_', 'Positive_']

ax = alluvial.plot(input_data,
                   colors = colors,
                   src_label_override = src_label_override,
                   dst_label_override = dst_label_override, 
                   v_gap_frac=0.08)

fig = ax.get_figure()
fig.set_size_inches(5,5)
ax.text(0.1, 0, prot1, ha="center", va="top", transform=ax.transAxes)
ax.text(0.9, 0, prot2, ha="center", va="top", transform=ax.transAxes)
plt.show()

#%% # CHECK VALUES -> OK
totals = {cond: Counter(roi[cond] for roi in Resp_ROI_dict.values()) for cond in protocols}
print(totals)
