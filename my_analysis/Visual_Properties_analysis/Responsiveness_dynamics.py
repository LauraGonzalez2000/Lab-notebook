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

from utils_.General_overview_episodes import compute_high_arousal_cond
from utils_.my_math import calc_stats, plot_stats
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
#%% 
####################################################################################################
#################################### RESPONSIVENESS DYNAMICS #######################################
######################################## ACROSS PROTOCOLS ##########################################
####################################################################################################

def generate_Resp_ROI_dict(data_s, protocols=[''], metric = "category", state='all', subprotocols=False):

    #initialize
    nROIS = sum(data.nROIs for data in data_s)
    protocols = protocols = [p for p in data_s[0].protocols if (p != 'grey-20min')]
    #Resp_ROI_dict = {f"ROI_{i}": dict.fromkeys(protocols, None) for i in range(nROIS)}
    Resp_ROI_dict = {f"ROI_{i}": {p: [] for p in protocols} for i in range(nROIS)}
    print(Resp_ROI_dict)
    
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
            print("varied params ", varied_params)

            param_values = []
            cond_p = ep.find_episode_cond()

            if len(varied_params) > 0 : 
                param_values = ep.varied_parameters[varied_params[0]]
                print("params values ", param_values)
                for param in varied_params:
                    cond_p = [ep.find_episode_cond(key=param,value=param_v) for param_v in param_values]
                    print(cond_p)
                
            print("range ", range(len(param_values)))
            print(cond_p)
            print(cond_b)
            if subprotocols==True: 

                if len(varied_params) > 0 : 
                    cond = [cond_p_i & cond_b for cond_p_i in cond_p]
                else: 
                    cond = [cond_p & cond_b]
            else: 
                cond=[cond_b]

            print("protocol ", p, " condition : \n", cond)

            for cond_i in cond: 
                for roi_n in range(data.nROIs):

                    t0 = max([0, ep.time_duration[0]-1.5])
                    
                    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                            interval_post=[t0, t0+1.5],                                   
                                            test='ttest', 
                                            sign='both')
                    if p== "looming-stim":
                        t0 = max([0, ep.time_duration[0]-0.5])
                        stat_test_props = dict(interval_pre=[-0.5,0],                                   
                                                interval_post=[t0, t0+0.5],                                   
                                                test='ttest', 
                                                sign='both')
                    
                    print("cond_i", cond_i)
                    roi_summary_data = pre_post_statistics(ep,
                                                    episode_cond = cond_i, #ep.find_episode_cond(),
                                                    response_args = dict(roiIndex=roi_n),
                                                    response_significance_threshold=0.05,
                                                    stat_test_props=stat_test_props,
                                                    repetition_keys=list(ep.varied_parameters.keys()), 
                                                    nMin_episodes=2)  #is that ok??
                    
                    raw_value = roi_summary_data["value"]
                    
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
    
    print(Resp_ROI_dict)

    return Resp_ROI_dict

def generate_behav_corr_ROI_dict(data_s, protocols=[''], subprotocols=False):

    #initialize
    nROIS = sum(data.nROIs for data in data_s)
    protocols = protocols = [p for p in data_s[0].protocols if (p != 'grey-20min')]
    #Resp_ROI_dict = {f"ROI_{i}": dict.fromkeys(protocols, None) for i in range(nROIS)}
    behav_corr_ROI_dict = {f"ROI_{i}": {p: [] for p in protocols} for i in range(nROIS)}
    print("before", behav_corr_ROI_dict)
    
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

            #print(running_speed.shape)
            #print(cond.shape)
           
            if subprotocols==True: 

                if len(varied_params) > 0 : 
                    param_values = ep.varied_parameters[varied_params[0]]
                    #print("params values ", param_values)
                    for param in varied_params:
                        cond_ = [ep.find_episode_cond(key=param,value=param_v) for param_v in param_values]
                        print(cond_)
                    cond = [cond_p_i for cond_p_i in cond_]
                else: 
                    cond = [cond]

            else: 
                cond=[cond]

            print("protocol ", p, " condition : \n", cond)

          
            
            for cond_i in cond : 

                print(cond_i)
                print("running_speed\n", running_speed.shape)
                print("cond_i\n", cond_i.shape)
                running_speed_i = running_speed[cond_i]
                print("running_speed i \n", running_speed_i.shape)

                for roi_n in range(data.nROIs):

                    #dFoF_i = dFoF[:][roi_n][:]
                    dFoF_i = dFoF[cond_i, roi_n, :]

                    
                    print("dFoF\n", dFoF_i.shape)
                    
                    corrcoef = np.corrcoef(running_speed_i, dFoF_i)
                    r_trials = [
                        np.corrcoef(running_speed_i[i], dFoF_i[i])[0, 1]
                        for i in range(dFoF_i.shape[0])
                    ]

                    r_mean = np.mean(r_trials)

                    print("corr coeff", corrcoef.shape)

                    behav_corr_ROI_dict[f"ROI_{nROI_id + roi_n}"][p].append(r_mean)

        nROI_id += data.nROIs
    
    #print(behav_corr_ROI_dict)

    return behav_corr_ROI_dict

def generate_input_data(Resp_ROI_dict, prot1, prot2, categories=("Positive", "NS", "Negative")):
    input_data = {src: {dst + "_": 0 for dst in categories} for src in categories}
    for ROI, responses in Resp_ROI_dict.items():
        src = responses[prot1]
        dst = responses[prot2] + "_"
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

cmap_graywarm = mcolors.LinearSegmentedColormap.from_list("graywarm",
                                                          ["#3b4cc0",  # blue (negative)
                                                           "#bdbbbb",  # mid gray (zero)
                                                           "#b40426"],   # red (positive)
                                                          N=256)

#%%
if __name__ == "__main__":
    #%%
    datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
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

    protocols = ["static-patch", 
                "drifting-gratings", 
                "Natural-Images-4-repeats", 
                "moving-dots", 
                "random-dots", 
                "looming-stim"]


    #%% CATEGORICAL DATA
    Resp_ROI_dict_c_all = generate_Resp_ROI_dict(data_s, metric="category", state='all')
    #Resp_ROI_dict_c_act = generate_Resp_ROI_dict(data_s, metric="category", state='active')
    #Resp_ROI_dict_c_rest = generate_Resp_ROI_dict(data_s, metric="category", state='rest')

    #%% LOAD DATA
    Resp_ROI_dict = Resp_ROI_dict_c_all
    #%% ALLUVIAL 
    prot1 = "drifting-gratings"
    prot2 = "looming-stim"
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




    #%% CONTINOUS DATA - NO SUBPROTOCOLS
    Resp_ROI_dict_v_all = generate_Resp_ROI_dict(data_s, metric="value", state="all", subprotocols=False)
    Resp_ROI_dict_v_act = generate_Resp_ROI_dict(data_s,  metric="value", state="active", subprotocols=False)
    Resp_ROI_dict_v_rest = generate_Resp_ROI_dict(data_s,  metric="value", state="rest", subprotocols=False)
    #%% CONTINOUS DATA - SUBPROTOCOLS
    Resp_ROI_dict_v_all_ = generate_Resp_ROI_dict(data_s, metric="value", state="all", subprotocols=True)
    Resp_ROI_dict_v_act_ = generate_Resp_ROI_dict(data_s,  metric="value", state="active", subprotocols=True)
    Resp_ROI_dict_v_rest_ = generate_Resp_ROI_dict(data_s,  metric="value", state="rest", subprotocols=True)

    #%% LOAD DATA
    Resp_ROI_dict = Resp_ROI_dict_v_all_
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

    #mapping = {'Positive': vmax, 'Negative': vmin, 'NS': 0}
    #df_numeric = df.replace(mapping)

    #df = df.sample(n=70) #zoom
    #df = df.sort_values(by="looming-stim", ascending=False)

    # ROI response vs STIM 
    fig, AX = pt.figure(figsize=(5,10), 
                        ax_scale=(2, 10)) 
            
    cmap_graywarm_nl = nonlinear_cmap(cmap_graywarm, vmin=vmin, vmax=vmax, exp = 0.7, N=256)

    print(np.max(df.values))
    print(np.min(df.values))
    print(df.values[7][0])

    print(df)

    AX.imshow(df.values, 
            aspect='auto', 
            cmap= cmap_graywarm_nl, 
            vmin = vmin,
            vmax = vmax)

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

    #%% STIM vs STIM similarity
    # Convert to matrix
    df = pd.DataFrame.from_dict(Resp_ROI_dict).T

    expanded_cols = []

    for col in df.columns:
        expanded = df[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df = pd.concat(expanded_cols, axis=1)

    # Compute stimulus × stimulus similarity (dot product across ROIs
    stim_similarity = np.ones((len(df.columns), len(df.columns)))*np.nan

    for i in range(stim_similarity.shape[0]):
        for j in range(stim_similarity.shape[0]):
            
            stim_similarity[i,j] = df[df.columns[i]].corr(df[df.columns[j]])

    pt.set_style("manuscript")

    # Plot heatmap
    fig, AX = pt.figure(figsize=(10,10),ax_scale=(2, 3.5) )
    vmin = -1 #np.min(stim_similarity.values)
    vmax = 1 #np.max(stim_similarity.values)
    AX.imshow(stim_similarity, 
            aspect='auto', 
            cmap= pt.plt.cm.PiYG, 
            vmin =vmin, 
            vmax=vmax)


    cmap_PiGY_nl = nonlinear_cmap(pt.plt.cm.PiYG, vmin=vmin, vmax=vmax, exp = 0.7, N=256)


    pt.bar_legend(AX,
                colorbar_inset=dict(rect=[1.1,.1,.04,.8]),
                colormap = cmap_PiGY_nl, #colormap=pt.binary, #pt.plt.cm.plasma #pt.plt.cm.coolwarm
                bar_legend_args={"fontsize":10},
                bounds=[vmin, vmax],
                ticks = [vmin, 0, vmax],
                #bar_legend_args={'size':2}, 
                label='Cross-correlation \nsimilarity')
    #              no_ticks=True)

    pt.set_plot(AX, 
                spines = ['bottom', 'left'],
                yticks=range(len(df.columns)), 
                yticks_labels=df.columns,
                xticks=range(len(df.columns)), 
                xticks_labels=df.columns,
                xticks_rotation=90,
                fontsize=5)

    for i in [-0.5, 1.5, 5.5, 7.5, 8.5, 13.5, 17.5]:
        print(i)
        AX.axvline(x=i, color='black', linewidth=0.5)
        AX.axhline(y=i, color='black', linewidth=0.5)

    #%%  Matrix of pairwise correlation between rest and run visual stimuli response vectors

    ########################################################################################
    # TEST GROUP
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
                    'values': np.ones((len(df_act.columns), len(df_act.columns)))*np.nan}


    for i in range(stim_similarity['values'].shape[0]):
        for j in range(stim_similarity['values'].shape[0]):
            if i==j:
                print("corr : ", df_act.columns[i], df_rest.columns[j])
                stim_similarity['corr'].append([f'{df_act.columns[i]}', f'{df_rest.columns[j]}'])
            stim_similarity['values'][i,j] = df_act[df_act.columns[i]].corr(df_rest[df_rest.columns[j]])

            #stim_similarity[i,j] = np.corrcoef(df[df.columns[i]], df[df.columns[j]])[0,1]

    print(stim_similarity)

    pt.set_style("manuscript")

    # Plot heatmap
    fig, AX = pt.figure(figsize=(10,10),ax_scale=(2, 3.5) )
    vmin = -1 #np.min(stim_similarity.values)
    vmax = 1 #np.max(stim_similarity.values)
    AX.imshow(stim_similarity['values'], 
            aspect='auto', 
            cmap= pt.plt.cm.PiYG, 
            vmin =vmin, 
            vmax=vmax)


    cmap_PiGY_nl = nonlinear_cmap(pt.plt.cm.PiYG, vmin=vmin, vmax=vmax, exp = 0.7, N=256)


    pt.bar_legend(AX,
                colorbar_inset=dict(rect=[1.1,.1,.04,.8]),
                colormap = cmap_PiGY_nl, #colormap=pt.binary, #pt.plt.cm.plasma #pt.plt.cm.coolwarm
                bar_legend_args={"fontsize":10},
                bounds=[vmin, vmax],
                ticks = [vmin, 0, vmax],
                #bar_legend_args={'size':2}, 
                label='Cross-correlation \nsimilarity')
    #              no_ticks=True)

    pt.set_plot(AX, 
                spines = ['bottom', 'left'],
                yticks=range(len(df.columns)), 
                yticks_labels=df.columns,
                xticks=range(len(df.columns)), 
                xticks_labels=df.columns,
                xticks_rotation=90,
                fontsize=5)

    for i in [-0.5, 1.5, 5.5, 7.5, 8.5, 13.5, 17.5]:
        print(i)
        AX.axvline(x=i, color='black', linewidth=0.5)
        AX.axhline(y=i, color='black', linewidth=0.5)

    #%%
    ########################################################################################
    ########################################################################################
    ########################################################################################
    # CONTROL GROUP - columns shuffled

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
                print("corr : ", df_act.columns[i], df_rest.columns[j])
                stim_similarity_control['corr'].append([f'{df_act.columns[i]}', f'{df_rest.columns[j]}'])
            stim_similarity_control['values'][i,j] = df_act[df_act.columns[i]].corr(df_rest[df_rest.columns[j]])

    #PLOT
    pt.set_style("manuscript")
    # Plot heatmap
    fig, AX = pt.figure(figsize=(10,10),ax_scale=(2, 3.5) )
    vmin = -1 #np.min(stim_similarity.values)
    vmax = 1 #np.max(stim_similarity.values)
    AX.imshow(stim_similarity_control['values'], 
            aspect='auto', 
            cmap= pt.plt.cm.PiYG, 
            vmin =vmin, 
            vmax=vmax)

    cmap_PiGY_nl = nonlinear_cmap(pt.plt.cm.PiYG, vmin=vmin, vmax=vmax, exp = 0.7, N=256)

    pt.bar_legend(AX,
                colorbar_inset=dict(rect=[1.1,.1,.04,.8]),
                colormap = cmap_PiGY_nl, #colormap=pt.binary, #pt.plt.cm.plasma #pt.plt.cm.coolwarm
                bar_legend_args={"fontsize":10},
                bounds=[vmin, vmax],
                ticks = [vmin, 0, vmax],
                #bar_legend_args={'size':2}, 
                label='Cross-correlation \nsimilarity')
    #              no_ticks=True)

    pt.set_plot(AX, 
                spines = ['bottom', 'left'],
                yticks=range(len(df_act.columns)), 
                yticks_labels=df_act.columns,
                xticks=range(len(df_rest.columns)), 
                xticks_labels=df_rest.columns,
                xticks_rotation=90,
                fontsize=5)

    ###########################################################################
    # plot correlation between same stimuli for rest and run visual stimuli response vectors (diagonal of the previous matrices)

    diag_control = np.diag(stim_similarity_control['values'])
    print(stim_similarity_control['corr'])
    print(diag_control)

    diag_test = np.diag(stim_similarity['values'])
    print(stim_similarity['corr'])
    print(diag_test)

    fig, AX = pt.figure(figsize=(5,5), 
                        ax_scale=(2, 3)) 

    labels_ = [ 'rest vs run similarity', 'control']

    AX.boxplot(x=[diag_control, diag_test], 
            tick_labels=["chance", "rest vs run\n similarity"], 
            widths = 0.6)

    stats = calc_stats("My title ", diag_control, diag_test, debug=True)
    plot_stats(ax=AX, n_groups = 2, stats=stats)

    n = len(diag_control)

    AX.scatter(np.ones(n), diag_control)      # chance at x = 1
    AX.scatter(np.ones(n) * 2, diag_test)    # real at x = 2

    for i in range(n):
        corr_group = stim_similarity['corr'][i]
        value_test = diag_test[i]
        for k in range(n):
            if stim_similarity_control['corr'][k][0] == corr_group[0]:
                value_control = diag_control[k]
                break;
        AX.plot([1, 2], [value_control, value_test], alpha=0.3, c="black")

    pt.set_plot(ax=AX, xticks = [1,2],
                yticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                ylim= [-0.3,1.2],
                ylabel='correlation', title='')












    #%% CORR COEFF for behavior for each cell - TO DO 
    ###########################################################################
    ###########################################################################

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

    df = df.sample(n=70) #zoom
    #df = df.sort_values(by="looming-stim", ascending=False)

    # ROI response vs STIM 
    fig, AX = pt.figure(figsize=(5,5), 
                        ax_scale=(2, 10)) 
            
    cmap_graywarm_nl = nonlinear_cmap(cmap_graywarm, vmin=vmin, vmax=vmax, exp = 0.7, N=256)

    AX.imshow(df.values, 
            aspect='auto', 
            cmap= cmap_graywarm_nl, 
            vmin = vmin,
            vmax = vmax)

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

