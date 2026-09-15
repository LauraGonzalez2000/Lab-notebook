# %% [markdown]
# # Responsiveness dynamics - Alluvial plot

#%%
# PACKAGES
import sys, os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

sys.path += ['../']
import utils_.alluvial_plot as alluvial 


#%%
# FUNCTIONS
def compute_high_arousal_cond(episodes, 
                              pre_stim = 0,
                              pupil_threshold = 0.29, 
                              running_speed_threshold = 0.1, 
                              metric = None):
    """
    Calculates wether the episodes are aroused/active or calm/resting.

    Args:
        episodes (array of Episode): (Episode#, ROI#, dFoF_values (0.5ms sampling rate)).
        pupil_threshold (float) : The threshold to discriminate calm state and aroused state
        running_speed_threshold (float): The threshold to discriminate resting state and active state.
        metric (string) : metric used to split calm/rest and aroused/active states. ("pupil" or "locomotion")

    Returns:
        np.array : HMcond is True when active/aroused and false when resting/calm
    """
    cond = []
    
    if metric=="pupil":
        '''
        if pupil_threshold is not None:
            cond = (episodes.pupil_diameter.mean(axis=1)>pupil_threshold)
        else:
            print("pupil_threshold not given")
        '''
        if pupil_threshold is not None: 
            start = int(pre_stim*1000)
            end = int(start + episodes.time_duration[0]*1000)
            values = episodes.pupil[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > pupil_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("pupil_threshold not given")
            


    if metric=="locomotion":
        
        if running_speed_threshold is not None: 
            start = int(pre_stim*1000)
            end = int(start + episodes.time_duration[0]*1000)
            values = episodes.running[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > running_speed_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("running_speed_threshold not given")

    return cond

def generate_Resp_ROI_dict(data_s, protocols=[''], metric = "category", state='all', subprotocols=False):

    #initialize
    nROIS = sum(data.nROIs for data in data_s)

    if protocols == ['']:
        protocols = [p for p in data_s[0].protocols if (p != 'grey-20min')]
    #Resp_ROI_dict = {f"ROI_{i}": dict.fromkeys(protocols, None) for i in range(nROIS)}
    Resp_ROI_dict = {f"ROI_{i}": {p: [] for p in protocols} for i in range(nROIS)}
    
    #fill
    nROI_id = 0
    for data in data_s:
        print("\n\n data : ", data, "\n\n")
        if protocols == ['']:
            protocols = [p for p in data.protocols if (p != 'grey-20min')]

        for p in protocols: 

            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF', 'running'])
            
            if state == 'all':
                cond_b = ep.find_episode_cond()
            elif state == "active":
                cond_b = compute_high_arousal_cond(ep, pre_stim=1, running_speed_threshold=0.1, metric="locomotion")
            elif state == "rest":
                cond_b = ~compute_high_arousal_cond(ep, pre_stim=1, running_speed_threshold=0.1, metric="locomotion")

            varied_params = [k for k in ep.varied_parameters.keys() if k != 'repeat']
            #varied_params = [ep.varied_parameters.keys()]

            print("varied params : ", varied_params)
       
            param_values = []
            cond_p = ep.find_episode_cond()

            if len(varied_params) > 0 : 
                print("ep.varied_parameters :", ep.varied_parameters)
                #param_values = ep.varied_parameters[[varied_param[0] for varied_param in varied_params]]
                param_values = [
                    v
                    for k, v in ep.varied_parameters.items()
                    if k != "repeat"
                ]

                print("param_values : ", param_values)
                #[array([ 0., 90.]), array([0.05      , 0.18571429, 0.32142857, 0.45714286, 0.59285714,
                #    0.72857143, 0.86428571, 1.        ])]
                

                #cond_p = []
                #for i, param in enumerate(varied_params): 
                #    for subvalue in param_values[i]:
                #        print(param, subvalue)
                #        cond_i = ep.find_episode_cond(key=param, value=subvalue)
                #        cond_p.append(cond_i)

                ####
                from itertools import product
                from functools import reduce

                values = [ep.varied_parameters[p] for p in varied_params]

                cond_p = []

                for combo in product(*values):
                    masks = [
                        ep.find_episode_cond(key=p, value=v)
                        for p, v in zip(varied_params, combo)
                    ]

                    cond = reduce(np.logical_and, masks)

                    cond_p.append(cond)

                    print("Here : ",dict(zip(varied_params, combo)))

                #for i, param in enumerate(varied_params):
                    #print(i, param)
                #    cond_p.append([ep.find_episode_cond(key=param, value=param_v) for param_v in param_values[i]])
                
                print("cond p : ",cond_p)

            if subprotocols==True: 
                if len(varied_params) > 0 : 
                    print("len cond p ", len(cond_p))

                    cond = [cond_p_i & cond_b for cond_p_i in cond_p]
                else: 
                    cond = [cond_p & cond_b]
            else: 
                cond=[cond_b]

            print("cond : ", cond)
            print("len cond : ", len(cond))

            for cond_i in cond: 
                for roi_n in range(data.nROIs):

                    t0 = max([0, ep.time_duration[0]-1.5])
                    
                    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                            interval_post=[t0, t0+1.5],                                   
                                            test='ttest', 
                                            sign='both')
                    if p == "looming-stim":
                        t0 = max([0, ep.time_duration[0]-0.5])
                        stat_test_props = dict(interval_pre=[-0.5,0],                                   
                                                interval_post=[t0, t0+0.5],                                   
                                                test='ttest', 
                                                sign='both')
                
                    roi_summary_data = pre_post_statistics(ep,
                                                    episode_cond = cond_i, #ep.find_episode_cond(),
                                                    response_args = dict(index=roi_n),
                                                    response_significance_threshold=0.05,
                                                    stat_test_props=stat_test_props,
                                                    repetition_keys=list(ep.varied_parameters.keys()), 
                                                    nMin_episodes=2)  #is that ok??
                    
                    raw_value = roi_summary_data["value"]
                    print("raw value ! :", raw_value)
                    
                    if raw_value: 
                        if isinstance(raw_value, (list, np.ndarray)):
                            value = float(np.array(raw_value).squeeze())
                        else:
                            value = float(raw_value)

                        #value = roi_summary_data['value']

                        if bool(roi_summary_data['significant'])==False:
                            category = 'NS'
                        else: 
                            if roi_summary_data['value']>0:
                                category = "Positive"
                            else: 
                                category = "Negative"

                        if metric == "category" : 
                            Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p].append(category)

                        elif metric == "value" : 
                            #Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p] = value
                            #print(Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p])
                            Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p].append(value)

        nROI_id += data.nROIs

    return Resp_ROI_dict

def generate_input_data(Resp_ROI_dict, prot1, prot2, categories=("Positive", "NS", "Negative"), subprot1=0, subprot2=0):
    input_data = {src: {dst + "_": 0 for dst in categories} for src in categories}
    print("input data before", input_data)
    for ROI, responses in Resp_ROI_dict.items():
        print("responses 1:",responses[prot1][subprot1])
        print("responses 2:",responses[prot2][subprot2])
        src = responses[prot1][subprot1]
        dst = responses[prot2][subprot2] + "_"
        input_data[src][dst] += 1
    return input_data

pt.set_style("manuscript")

#%% 
###### COMPARE PROTOCOLS ################################################

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
    data.build_running()
    data.build_facemotion()
    data.build_pupil()
    data_s.append(data)
#%%
#protocols = ["static-patch", 
#            "drifting-gratings", 
#            "Natural-Images-4-repeats"]

protocols = ["moving-dots",
             "random-dots",
             "static-patch",
             "looming-stim", 
             "Natural-Images-4-repeats", 
             "drifting-gratings"]

#%% GENERATE CATEGORICAL DATA DICT
Resp_ROI_dict_c_all = generate_Resp_ROI_dict(data_s, metric="category", state='all')
#Resp_ROI_dict_c_act = generate_Resp_ROI_dict(data_s, metric="category", state='active')
#Resp_ROI_dict_c_rest = generate_Resp_ROI_dict(data_s, metric="category", state='rest')

#%% LOAD THE DESIRED DATA (_all , _act, _rest)
Resp_ROI_dict = Resp_ROI_dict_c_all
#%% PLOT ALLUVIAL
# Choose desired pair 

#prot1 = "static-patch"
#prot2 = "drifting-gratings"
#--------------------------------
#prot1="drifting-gratings"
#prot2="Natural-Images-4-repeats"
#---------------------------------
#prot1="Natural-Images-4-repeats"
#prot2="moving-dots"
#---------------------------------
#prot1="moving-dots"
#prot2="random-dots"
#---------------------------------
#prot1="random-dots"
#prot2="looming-stim"
#---------------------------------
#prot1="looming-stim"
#prot2="static-patch"
#--------------------------------
prot1="Natural-Images-4-repeats"
prot2="static-patch"

input_data = generate_input_data(Resp_ROI_dict, prot1, prot2)

colors = ["#3b4cc0", "#bdbbbb", "#b40426"]
src_label_override=["Negative", 'NS', 'Positive']
dst_label_override=["Negative_", 'NS_', 'Positive_']

ax = alluvial.plot(input_data,
                colors = colors,
                src_label_override = src_label_override,
                dst_label_override = dst_label_override, 
                h_gap_frac=0.03,
                v_gap_frac=0.2)

fig = ax.get_figure()
fig.set_size_inches(5,5)
ax.text(0.1, -0.2, prot1, ha="center", va="top", transform=ax.transAxes)
ax.text(0.9, -0.2, prot2, ha="center", va="top", transform=ax.transAxes)
plt.show()


################### COMPARE SUBPROTOCOLS #####################################
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'NWBs_8contrasts2ori')
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
    data.build_running()
    data.build_facemotion()
    data.build_pupil()
    data_s.append(data)

protocols = ["ff-gratings-2orientations-8contrasts-15repeats"]

#%%
Resp_ROI_dict_c_all = generate_Resp_ROI_dict(data_s, metric="category", state='all', subprotocols=True)
#%%
Resp_ROI_dict = Resp_ROI_dict_c_all
#%%
prot1="ff-gratings-2orientations-8contrasts-15repeats"
prot2="ff-gratings-2orientations-8contrasts-15repeats"
subprot1=2
subprot2=5

input_data = generate_input_data(Resp_ROI_dict, prot1, prot2, subprot1=subprot1, subprot2=subprot2)

colors = ["#3b4cc0", "#bdbbbb", "#b40426"]
src_label_override=["Negative", 'NS', 'Positive']
dst_label_override=["Negative_", 'NS_', 'Positive_']

ax = alluvial.plot(input_data,
                colors = colors,
                src_label_override = src_label_override,
                dst_label_override = dst_label_override, 
                h_gap_frac=0.03,
                v_gap_frac=0.2)

fig = ax.get_figure()
fig.set_size_inches(5,5)
ax.text(0.1, -0.2, prot1, ha="center", va="top", transform=ax.transAxes)
ax.text(0.9, -0.2, prot2, ha="center", va="top", transform=ax.transAxes)
plt.show()
#%%