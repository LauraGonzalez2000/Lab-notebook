# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import os, sys
import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

sys.path += ['..']
from utils_.General_overview_episodes import compute_high_arousal_cond

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

pt.set_style('manuscript')

from scipy.cluster.hierarchy import optimal_leaf_ordering
from matplotlib.transforms import blended_transform_factory

from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


#%%  FUNCTIONS
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

def plot_evoked_pattern(EP_s,  
                        pattern_cond = [], 
                        quantity='rawFluo',
                        with_stim_inset=True,
                        axR=None, 
                        behavior_split=False, 
                        clustering = 'PCA'):

    if with_stim_inset and (ep_s[0].visual_stim is None):
        print('\n [!!] visual stim of episodes was not initialized  [!!]  ')
        print('    --> screen_inset display desactivated ' )
        with_stim_inset = False

    # SET FIGURE #################################################################
    n_stim = len(np.unique(EP_s[0].index)) #assumes all files have same stimuli
    if not behavior_split:
        n_cond = n_stim
    elif behavior_split: 
        n_cond = n_stim*2 #columns are doubled 

    #Calculate behavior condition for each file :
    HMcond_s = []
    for i, ep in enumerate(EP_s):
        HMcond = compute_high_arousal_cond(ep, pre_stim=1, running_speed_threshold=0.1, metric="locomotion")
        HMcond = np.array(HMcond)

        if i>0 and len(HMcond)!=len(HMcond_s[0]):
            n_missing = len(HMcond_s[0]) - len(HMcond)
            pad_shape = (n_missing,)
            pad = np.full(pad_shape, False)
            HMcond = np.concatenate([HMcond, pad], axis=0)

        HMcond_s.append(HMcond)
    
    #Calculate pattern condition for each stimulus
    Patterncond_s = []
    for stim_id in range(n_stim):
        pattern_cond = np.array([ep_s[0].index[i] == stim_id for i in range(len(ep_s[0].index))])
        Patterncond_s.append(pattern_cond)

    ####### initialize figure
    fig, axR = pt.figure(axes=(n_cond,1),
                         ax_scale=(2, 11),
                         right=4,
                         left=0.3,
                         top=(4.5 if with_stim_inset else 1))
    
    resp_s = []
    for file_i in range(len(EP_s)):  #for each file
        resp = np.array(getattr(EP_s[file_i], quantity))
        
        if resp.shape[0] != EP_s[0].dFoF.shape[0]:
            print(f"In file {file_i} some trials are missing : {resp.shape[0]} instead of {EP_s[0].dFoF.shape[0]}")
            #pad missing trials with NaNs
            n_missing = EP_s[0].dFoF.shape[0] - resp.shape[0]
            pad_shape = (n_missing,) + resp.shape[1:]
            pad = np.full(pad_shape, np.nan)
            resp = np.concatenate([resp, pad], axis=0)

        for stim_id in range(n_stim):
            if not behavior_split:
                pattern_cond = Patterncond_s[stim_id]
                temp = resp[pattern_cond,:,:]
                resp_s.append(temp)
            
            elif behavior_split:
                HMcond = HMcond_s[file_i]
                pattern_cond = Patterncond_s[stim_id]

                final_cond = HMcond & pattern_cond
                temp = resp[final_cond,:,:]
                resp_s.append(temp)

                final_cond2= ~HMcond & pattern_cond                
                temp2 = resp[final_cond2,:,:]
                resp_s.append(temp2)

    column_lists = [[] for _ in range(n_cond)] # column[i] will store the responses of this specific stimuli with behavioral condition if needed
    
    for i, resp in enumerate(resp_s):
        stim_idx = i % n_cond     
        column_lists[stim_idx].append(resp)

    varied_params = list(ep_s[0].varied_parameters.keys())[0]
    param_values = ep_s[0].varied_parameters[varied_params]
    order_mantained = []

    for column in range(n_cond):
        if behavior_split==False:
            stim_idx = column
            axR[column].set_title(f"STIM {stim_idx+1}", y=1.3, fontsize=15)

        elif behavior_split==True:
            stim_idx = column // 2
            state = "ACT" if column % 2 == 0 else "REST"
            axR[column].set_title(f"STIM {stim_idx+1} {state}", y=1.3, fontsize=15)
        
        print(" Column : ", column, "stim idx", stim_idx) 
        # STIM INSET ##################################################################
        param = param_values[stim_idx]
        stim_inset = pt.inset(axR[column], [0.1, 0.8, 0.8, 0.8])
        cond = ep.find_episode_cond(key=varied_params,value=param)
        iStim = np.flatnonzero(cond)[0]
        image =  ep.visual_stim.get_image(iStim)
        image =  np.rot90(image, k=1)
        stim_inset.imshow(image, cmap=pt.plt.cm.binary_r,vmin=0, vmax=1)
        stim_inset.axis('off')
        
        # RASTER ######################################################################
        # mean response for raster
        resp_ = column_lists[column]
        mean_resp_s = []
        for r in resp_:
            if r is None or r.shape[0] == 0: #no trials for this file
                mean_resp_s.append(np.full((r.shape[1], r.shape[2]), np.nan)) #or...not plot invalid rois for the other state
            else:
                mean_resp = np.nanmean(r, axis=0) #mean over trials!
                mean_resp_s.append(mean_resp)

        combined = np.concatenate(mean_resp_s, axis=0)

        #subtract baseline
        combined_zerobaseline = np.asarray([trace - np.mean(trace[ 0 : int(-(ep_s[0].t[0])*1000)]) for trace in combined])
        valid = np.isfinite(combined_zerobaseline).all(axis=1) & (np.nanstd(combined_zerobaseline, axis=1) > 0)

        #reorder neurons by similarity (but keep same order between act and rest)
        if clustering == "corr_link":
            #old option -> more for discrete clusters, dendograms 
            dist = pdist(combined_zerobaseline[valid], metric='correlation')
            Z = linkage(dist, method="weighted") #average #complete
            Zopt = optimal_leaf_ordering(Z, dist)
            if not behavior_split: 
                order = leaves_list(Zopt)
            if behavior_split and state == "ACT":
                order = leaves_list(Zopt)
            elif behavior_split and state== "REST":
                order = order_mantained #not recalculated, taking the previous one (ACT of the same stim)
            combined_zerobaseline_ordered = combined_zerobaseline[order, :]
            order_mantained = order


        if clustering == "PCA": 
            #PCA -> more for smooth gradients, temporal motifs
            X = combined_zerobaseline[valid]
            Xz = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
            pca = PCA(n_components=1)
            pc1_scores = pca.fit_transform(Xz).squeeze()
            pca_order_valid = np.argsort(-pc1_scores)
            valid_indices = np.where(valid)[0]
            if not behavior_split: 
                order = valid_indices[pca_order_valid]
            if behavior_split and state == "ACT":
                order = valid_indices[pca_order_valid]
            elif behavior_split and state== "REST":
                order = order_mantained #not recalculated, taking the previous one (ACT of the same stim)
            combined_zerobaseline_ordered = combined_zerobaseline[order, :]
            order_mantained = order

        if clustering == 'amplitude':
            # -> more intuitive
            X = combined_zerobaseline[valid]
            #takes las second of stimulation
            t_start = int(ep_s[0].time_duration[0] * 1000) #assumes 1s prestim
            t_end   = int((ep_s[0].time_duration[0] + 1) * 1000) #assumes 1 s prestim
            amplitudes = np.mean(X[:, t_start:t_end], axis=1)
            amp_order_valid = np.argsort(-amplitudes)
            valid_indices = np.where(valid)[0]
            if not behavior_split:
                order = valid_indices[amp_order_valid]
            if behavior_split and state == "ACT":
                order = valid_indices[amp_order_valid]
            elif behavior_split and state == "REST":
                order = order_mantained  # keep ACT order

            combined_zerobaseline_ordered = combined_zerobaseline[order, :]
            order_mantained = order

        axR[column].axvline(0, linestyle='--', linewidth=0.5)
        axR[column].axvline(ep_s[0].time_duration[0], linestyle='--', linewidth=0.5)

        # Plot raster
        vmin = -1
        vmax = 3
        cmap_graywarm = mcolors.LinearSegmentedColormap.from_list( "graywarm",
                                                                   ["#3b4cc0",  # blue (negative)
                                                                   "#bdbbbb",  # mid gray (zero)
                                                                   "#b40426"   # red (positive)
                                                                   ],
                                                                   N=256)
        
        cmap_graywarm_nl = nonlinear_cmap(cmap_graywarm, vmin=vmin, vmax=vmax, exp = 0.7, N=256)

        axR[column].imshow(combined_zerobaseline_ordered,
                           cmap = cmap_graywarm_nl, #cmap=pt.binary, #pt.plt.cm.plasma #pt.plt.cm.coolwarm
                           aspect='auto', 
                           interpolation='none',
                           vmin = vmin,
                           vmax = vmax,
                           extent=(ep_s[0].t[0], ep_s[0].t[-1], 0, combined_zerobaseline_ordered.shape[0]))

        time_max = ep_s[0].time_duration[0] + 1 #assumaes prestim 1
        pt.set_plot(axR[column], 
                    spines = ['bottom'],
                    yticks=[0, len(combined_zerobaseline_ordered)],
                    ylabel='ROI',
                    xticks=np.arange(-1, time_max+1, 1), 
                    xlabel='Time (s)',
                    xlim=[ep_s[0].t[0], ep_s[0].t[-1]], 
                    fontsize=15)
       
        pt.bar_legend(axR[column], 
                      colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
                      colormap = cmap_graywarm_nl, #colormap=pt.binary, #pt.plt.cm.plasma #pt.plt.cm.coolwarm
                      bar_legend_args={'size':1},
                      label='$\\Delta$F/F',
                      X=np.arange(vmin, vmax+0.5, 0.5),
                      bounds=[vmin, vmax],
                      ticks = None,
                      ticks_labels=None,
                      no_ticks=False,
                      orientation='vertical', 
                      fontsize=15)
        
        trans = blended_transform_factory(axR[column].transData, axR[column].transAxes)

        axR[column].fill_between([0, ep_s[0].time_duration[0]],  # x in seconds (data coords)
                                 1.02, 1.06,                    # y in axes coords (above top)
                                 transform=trans,
                                 color="k",
                                 alpha=0.6,
                                 linewidth=0,
                                 clip_on=False)

        
        axR[column].annotate(f"{len(combined_zerobaseline_ordered)}", 
                             xy=(-0.25,0), 
                             fontsize=15,
                             xycoords="axes fraction",
                             clip_on=False)
        

    return fig

#%%
#LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
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

#%%
#protocols = ["static-patch",  "drifting-gratings", "Natural-Images-4-repeats"]
protocols = ["Natural-Images-4-repeats"]

ep_s_ = []
for protocol in protocols: 
    ep_s = []
    for i, data in enumerate(data_s): 
        print("File ", i)
        ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed', 'rawFluo'])
        ep.init_visual_stim(data) 
        ep_s.append(ep)
    ep_s_.append(ep_s)

#%% 
########################################################################
##################### RESULTS PER PROTOCOL #############################
########################################################################
for p, protocol in enumerate(protocols):
    ep_s = ep_s_[p]
    plot_evoked_pattern(EP_s=ep_s, 
                        quantity='dFoF', 
                        with_stim_inset=True, 
                        behavior_split=False, 
                        clustering = 'amplitude')
    
    plot_evoked_pattern(EP_s=ep_s, 
                        quantity='dFoF', 
                        with_stim_inset=True, 
                        behavior_split=True, 
                        clustering = 'amplitude')
#######################################################################################################################
#######################################################################################################################

#%% my data ##############################################################
##########################################################################
##########################################################################
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','vision-survey', 'NDNF-Cre-batch1','NWBs')
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

#%%
protocols = ["static-patch"]

ep_s_ = []
for protocol in protocols: 
    ep_s = []
    for i, data in enumerate(data_s): 
        print("File ", i)
        ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed', 'rawFluo'])
        ep.init_visual_stim(data) 
        ep_s.append(ep)
    ep_s_.append(ep_s)

#%%
for p, protocol in enumerate(protocols):
    ep_s = ep_s_[p]
    plot_evoked_pattern(EP_s=ep_s, 
                        quantity='dFoF', 
                        with_stim_inset=True, 
                        behavior_split=False)
    plot_evoked_pattern(EP_s=ep_s, 
                        quantity='dFoF', 
                        with_stim_inset=True, 
                        behavior_split=True)
