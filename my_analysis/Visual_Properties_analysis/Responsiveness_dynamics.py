# %% [markdown]
# # Responsiveness dynamics

#%%
# PACKAGES
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from pathlib import Path

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

sys.path += ['../']
from utils_.General_overview_episodes import compute_high_arousal_cond
from utils_.my_math import calc_stats, plot_stats
import utils_.alluvial_plot as alluvial


#%%
# FUNCTIONS
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

            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF', 'running_speed'])
            
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
                                                    response_args = dict(roiIndex=roi_n),
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

def plot_heatmap(stim_similarity, separators=True):
    # Plot heatmap
    fig, AX = pt.figure(figsize=(10,10),ax_scale=(2, 3.5))
    vmin = -1 #np.min(stim_similarity.values)
    vmax = 1 #np.max(stim_similarity.values)
    AX.imshow(stim_similarity, 
            aspect='auto', 
            cmap= pt.plt.cm.PiYG, 
            vmin =vmin, 
            vmax=vmax)

    cmap_PiGY_nl = nonlinear_cmap(pt.plt.cm.PiYG, 
                                  vmin=vmin, vmax=vmax, 
                                  exp = 0.7, N=256)

    pt.bar_legend(AX,
                colorbar_inset=dict(rect=[1.1,.1,.04,.8]),
                colormap = cmap_PiGY_nl, 
                           # pt.binary, 
                           # pt.plt.cm.plasma 
                           # pt.plt.cm.coolwarm
                bar_legend_args={"fontsize":10},
                bounds=[vmin, vmax],
                ticks = [vmin, 0, vmax],
                # bar_legend_args={'size':2}, 
                label='Cross-correlation \nsimilarity')
                # no_ticks=True)

    pt.set_plot(AX, 
                spines = ['bottom', 'left'],
                yticks=range(len(df.columns)), 
                yticks_labels=df.columns,
                xticks=range(len(df.columns)), 
                xticks_labels=df.columns,
                xticks_rotation=90,
                fontsize=5)
    if separators: 
        for i in [-0.5, 1.5, 5.5, 7.5, 8.5, 13.5, 17.5]:
            print(i)
            AX.axvline(x=i, color='black', linewidth=0.5)
            AX.axhline(y=i, color='black', linewidth=0.5)
    
cmap_graywarm = mcolors.LinearSegmentedColormap.from_list("graywarm",
                                                          ["#3b4cc0",  # blue (negative)
                                                           "#bdbbbb",  # mid gray (zero)
                                                           "#b40426"],   # red (positive)
                                                          N=256)

pt.set_style("manuscript")
#%%
if __name__ == "__main__":
    #%% 
    ####################################################################################################
    #################################### RESPONSIVENESS DYNAMICS #######################################
    ######################################## ACROSS PROTOCOLS ##########################################
    ########################################   CATEGORICAL    ##########################################
    ####################################################################################################

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
    protocols = ["static-patch", 
                "drifting-gratings", 
                "Natural-Images-4-repeats"]

    #protocols = ["moving-dots",
    #             "random-dots",
    #             "static-patch",
    #             "looming-stim", 
    #             "Natural-Images-4-repeats", 
    #             "drifting-gratings"]


    #####################################################################################################
    ##############################  ALLUVIAL PLOTS   ####################################################
    ############################## CATEGORICAL DATA  ####################################################
    #####################################################################################################
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
        data.build_running_speed()
        data.build_facemotion()
        data.build_pupil_diameter()
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
    ####################################################################################################
    #################################### RESPONSIVENESS DYNAMICS #######################################
    ######################################## ACROSS PROTOCOLS ##########################################
    ########################################   CONTINOUS      ##########################################
    ####################################################################################################
    #%% GENERATE CONTINOUS DATA DICT - NO SUBPROTOCOLS
    Resp_ROI_dict_v_all = generate_Resp_ROI_dict(data_s, 
                                                 metric="value", 
                                                 state="all", 
                                                 subprotocols=False)
    #Resp_ROI_dict_v_act = generate_Resp_ROI_dict(data_s,  
    #                                             metric="value", 
    #                                             state="active", 
    #                                             subprotocols=False)
    #Resp_ROI_dict_v_rest = generate_Resp_ROI_dict(data_s,  
    #                                              metric="value", 
    #                                              state="rest", 
    #                                              subprotocols=False)
    
    #%% GENERATE CONTINOUS DATA DICT - SUBPROTOCOLS
    Resp_ROI_dict_v_all_ = generate_Resp_ROI_dict(data_s, 
                                                  protocols=protocols, 
                                                  metric="value", 
                                                  state="all", 
                                                  subprotocols=True)
    Resp_ROI_dict_v_act_ = generate_Resp_ROI_dict(data_s,  
                                                  metric="value", 
                                                  state="active", 
                                                  subprotocols=True)
    Resp_ROI_dict_v_rest_ = generate_Resp_ROI_dict(data_s,  
                                                   metric="value", 
                                                   state="rest", 
                                                   subprotocols=True)

    #%% LOAD THE DESIRED DATA (_all , _act, _rest, _all_ , _act_, _rest_)
    Resp_ROI_dict = Resp_ROI_dict_v_all_
    #%%
    ######################################################################
    ##############################   RASTER PLOTS    #####################
    ##############################  CONTINOUS DATA   #####################
    ######################################################################
    #%% RASTER PLOT - PEAK AMPLITUDE
    vmin = -1
    vmax = 3

    # Convert to matrix
    df = pd.DataFrame.from_dict(Resp_ROI_dict).T

    expanded_cols = []

    for col in df.columns:
        expanded = df[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df = pd.concat(expanded_cols, axis=1)

    # ROI response vs STIM 
    fig, AX = pt.figure(figsize=(5,10), 
                        ax_scale=(3, 10)) 
            
    cmap_graywarm_nl = nonlinear_cmap(cmap_graywarm, 
                                      vmin=vmin, 
                                      vmax=vmax, 
                                      exp = 0.7, 
                                      N=256)

    AX.imshow(df.values, 
            aspect='auto', 
            cmap= cmap_graywarm_nl, 
            vmin = vmin,
            vmax = vmax, 
            interpolation='nearest')

    pt.bar_legend(AX, 
                colorbar_inset=dict(rect=[1.1,.1,.04,.8], 
                                    facecolor=None),
                colormap = cmap_graywarm_nl, 
                            #pt.binary, 
                            # #pt.plt.cm.plasma 
                            # #pt.plt.cm.coolwarm
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

    #%%
    ####################################################################
    #############               HEATMAP            #####################
    #############        STIM VS STIM SIMILARITY   #####################
    ####################################################################

    df = pd.DataFrame.from_dict(Resp_ROI_dict).T

    expanded_cols = []

    for col in df.columns:
        expanded = df[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df = pd.concat(expanded_cols, axis=1)

    stim_similarity = np.ones((len(df.columns), len(df.columns)))*np.nan

    for i in range(stim_similarity.shape[0]):
        for j in range(stim_similarity.shape[0]):
            stim_similarity[i,j] = df[df.columns[i]].corr(df[df.columns[j]])

    plot_heatmap(stim_similarity)

    #%%
    ####################################################################
    #############            HEATMAP                ####################
    #############              TEST                 ####################
    ############# STIM ACT VS STIM REST SIMILARITY  ####################
    ####################################################################

    # Convert to matrix
    df_act  = pd.DataFrame.from_dict(Resp_ROI_dict_v_act_).T
    df_rest = pd.DataFrame.from_dict(Resp_ROI_dict_v_rest_).T

    expanded_cols = []

    for col in df_act.columns:
        expanded = df_act[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df_act = pd.concat(expanded_cols, axis=1)

    expanded_cols = []

    for col in df_rest.columns:
        expanded = df_rest[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df_rest = pd.concat(expanded_cols, axis=1)

    # Compute stimulus × stimulus similarity (dot product across ROIs)
    stim_similarity = {'corr' : [], 
                       'values': np.ones((len(df_act.columns), 
                                          len(df_act.columns)))*np.nan}


    for i in range(stim_similarity['values'].shape[0]):
        for j in range(stim_similarity['values'].shape[0]):
            if i==j:
                #print("corr : ", df_act.columns[i], df_rest.columns[j])
                stim_similarity['corr'].append([f'{df_act.columns[i]}', 
                                                f'{df_rest.columns[j]}'])
            stim_similarity['values'][i,j] = df_act[df_act.columns[i]]\
                                            .corr(df_rest[df_rest.columns[j]])

            #stim_similarity[i,j] = np.corrcoef(df[df.columns[i]], df[df.columns[j]])[0,1]

    plot_heatmap(stim_similarity['values'])
    
    #%%
    ####################################################################
    #############            HEATMAP                ####################
    #############        CONTROL (SHUFFLE)          ####################
    ############# STIM ACT VS STIM REST SIMILARITY  ####################
    ####################################################################
    #%% Example
    # Convert to matrix
    df_act  = pd.DataFrame.from_dict(Resp_ROI_dict_v_act_).T
    df_rest = pd.DataFrame.from_dict(Resp_ROI_dict_v_rest_).T

    expanded_cols = []

    for col in df_act.columns:
        expanded = df_act[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df_act = pd.concat(expanded_cols, axis=1)
    df_act = df_act.sample(frac=1, axis=1)

    expanded_cols = []

    for col in df_rest.columns:
        expanded = df_rest[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df_rest = pd.concat(expanded_cols, axis=1)
    df_rest = df_rest.sample(frac=1, axis=1)

    # Compute stimulus × stimulus similarity (dot product across ROIs)
    stim_similarity_control = {'corr' : [], 
                            'values': np.ones((len(df_act.columns), len(df_act.columns)))*np.nan}
    for i in range(stim_similarity_control['values'].shape[0]):
        for j in range(stim_similarity_control['values'].shape[0]):
            if i==j:
                stim_similarity_control['corr'].append([f'{df_act.columns[i]}', f'{df_rest.columns[j]}'])
            stim_similarity_control['values'][i,j] = df_act[df_act.columns[i]].corr(df_rest[df_rest.columns[j]])
            
    plot_heatmap(stim_similarity_control['values'], separators=False)

    ###########################################################################
    # plot correlation between randomly associated stimuli 
    # for rest and run visual stimuli response vectors (diagonal of the previous matrix)
    diag_control = np.diag(stim_similarity_control['values']) #np.mean(diag_control_s, axis=0)
    diag_test = np.diag(stim_similarity['values'])

    fig, AX = pt.figure(figsize=(5,5), 
                        ax_scale=(1, 3)) 

    AX.boxplot(x=[diag_test, diag_control], 
               positions = [0,1],
                tick_labels=["rest \nvs run\n similarity", "chance"], 
                widths = 0.6)#, 
                #angle='vertical')

    stats = calc_stats("My title ", diag_test, diag_control,  debug=True)
    plot_stats(ax=AX, n_groups = 2, stats=stats)

    n = len(diag_control)

    AX.scatter(np.zeros(n) , diag_test)      # real at x = 1
    AX.scatter(np.ones(n) , diag_control)    # chance at x = 2
    
    for i in range(n):
        corr_group = stim_similarity['corr'][i]
        value_test = diag_test[i]
        for k in range(n):
            if stim_similarity_control['corr'][k][0] == corr_group[0]:
                value_control = diag_control[k]
                break;
        AX.plot([0, 1], [value_test, value_control], alpha=0.3, c="black")

    pt.set_plot(ax=AX, xticks = [0,1],
                yticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                ylim= [-0.3,1.2],
                ylabel='correlation', title='')


    #%% repeat it 100 times to have statistical confirmation?
    diag_control_s = []
    for k in range(100):
        # Convert to matrix
        df_act  = pd.DataFrame.from_dict(Resp_ROI_dict_v_act_).T
        df_rest = pd.DataFrame.from_dict(Resp_ROI_dict_v_rest_).T

        expanded_cols = []

        for col in df_act.columns:
            expanded = df_act[col].apply(pd.Series)
            expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
            expanded_cols.append(expanded)

        df_act = pd.concat(expanded_cols, axis=1)
        df_act = df_act.sample(frac=1, axis=1)

        expanded_cols = []

        for col in df_rest.columns:
            expanded = df_rest[col].apply(pd.Series)
            expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
            expanded_cols.append(expanded)

        df_rest = pd.concat(expanded_cols, axis=1)
        df_rest = df_rest.sample(frac=1, axis=1)

        # Compute stimulus × stimulus similarity (dot product across ROIs)
        stim_similarity_control = {'corr' : [], 
                                   'values': np.ones((len(df_act.columns), len(df_act.columns)))*np.nan}
        for i in range(stim_similarity_control['values'].shape[0]):
            for j in range(stim_similarity_control['values'].shape[0]):
                if i==j:
                    stim_similarity_control['corr'].append([f'{df_act.columns[i]}', f'{df_rest.columns[j]}'])
                stim_similarity_control['values'][i,j] = df_act[df_act.columns[i]].corr(df_rest[df_rest.columns[j]])
                diag_control = np.diag(stim_similarity_control['values'])

        diag_control_s.append(diag_control)
    
    #to something with diag_control -> plot histogram or something
