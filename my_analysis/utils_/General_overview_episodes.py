# %% [markdown]
# #General overview episodes
# Contains functions useful to do basic visualization

# %%
# load packages:
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Programming', 'Lab-notebook', 'physion', 'src'))
from physion.utils  import plot_tools as pt
import numpy as np
import itertools
from physion.analysis.episodes.build import EpisodeData

# %%
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
            values = episodes.pupil_diameter[:, start:end]  ## check if these boundaries cause problem #1000:3001
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
            values = episodes.running_speed[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > running_speed_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("running_speed_threshold not given")

    return cond

def get_trial_average_trace(episodes,
                            quantity='dFoF',
                            roiIndex=None,
                            condition=None,
                            with_std_over_rois=False):
    """
    Return trial-averaged response trace (mean and SEM) for one Episodes object.
    """

    if condition is None:
        condition = np.ones(np.sum(episodes.protocol_cond_in_full_data), dtype=bool)
    elif len(condition) == len(episodes.protocol_cond_in_full_data):
        condition = condition[episodes.protocol_cond_in_full_data]

    avg_dim = 'episodes' if with_std_over_rois else 'ROIs'

    response = episodes.get_response2D(quantity=quantity,
                                       episode_cond=condition,
                                       roiIndex=roiIndex,
                                       averaging_dimension=avg_dim)
    if response.size == 0:
        return None, None

    mean_trace = response.mean(axis=0)
    sem_trace  = response.std(axis=0) / np.sqrt(response.shape[0])

    return mean_trace, sem_trace

def plot_dFoF_per_protocol(data_s,
                           dataIndex=None,
                           roiIndex=None,
                           pupil_threshold=2.9,
                           running_speed_threshold=0.5, 
                           metric=None, 
                           protocols = [], 
                           subplots_n=5):
    """
    Plot dFoF per protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    roiIndex : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : str or None
        Metric to split high/low arousal conditions.
    """
    
    # select sessions
    if dataIndex is not None:
        mode = "single"
    else:
        mode = "average"
    

    fig, AX = pt.figure(axes_extents=[[ [1,1] for _ in protocols ] for _ in range(subplots_n)])  #generalize 9 

    for p, protocol in enumerate(protocols):
        session_traces = []

        for data in data_s:
            episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Running-Speed'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)

            if metric is not None:
                cond = compute_high_arousal_cond(episodes, pupil_threshold, running_speed_threshold, metric=metric)
            else:
                cond = episodes.find_episode_cond()
            
            varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
            varied_values = [episodes.varied_parameters[k] for k in varied_keys]

            i = 0
            for values in itertools.product(*varied_values):
                stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

                mean_trace, sem_trace = get_trial_average_trace(
                    episodes,
                    roiIndex=roiIndex,
                    condition=stim_cond & cond
                )
                
                if mean_trace is not None:
                    session_traces.append((i, mean_trace, sem_trace))
                i += 1
        
        # plotting
        n_conditions = len(list(itertools.product(*varied_values)))

        for j in range(n_conditions):
            traces = [tr for idx, tr, _ in session_traces if idx == j]
            sems   = [se for idx, _, se in session_traces if idx == j]
            
            if len(traces) == 0:
                continue  # nothing to plot for this condition

            if mode == "single":
                mean_trace = traces[0]
                sem_trace  = sems[0]
            else:
                mean_trace = np.mean(traces, axis=0)
                sem_trace  = np.std(traces, axis=0) / np.sqrt(len(traces))

            
            AX[j][p].plot(mean_trace, color='k')
            AX[j][p].fill_between(np.arange(len(mean_trace)),
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color='k', alpha=0.3)
            AX[j][p].axvspan(1000, 1000+1000*episodes.time_duration[0], color='lightgrey', alpha=0.5, zorder=0)

        AX[0][p].set_title(protocol.replace('Natural-Images-4-repeats','natural-images'))   
        #AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
        #                  (0.5,1.4),
        #                  xycoords='axes fraction', ha='center', fontsize=7)
    
    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            AX[-1][0].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (0, -0.2), xycoords='axes fraction')
        else:
            AX[-1][0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (0, -0.2), xycoords='axes fraction')
    else:
        if mode == "single":
            AX[-1][0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data_s[0].filename.replace('.nwb','')),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
        else:
            AX[-1][0].annotate('roi #%i , average over %i sessions' % (1+roiIndex, len(data_s)),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
    pt.set_common_xlims(AX)
    
    return fig, AX

def plot_dFoF_of_protocol(data_s,
                           dataIndex=None,
                           roiIndex=None,
                           pupil_threshold=2.9,
                           running_speed_threshold=0.5, 
                           metric=None, 
                           protocol = "", 
                           subplots_n=5):
    """
    Plot dFoF per protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    roiIndex : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : str or None
        Metric to split high/low arousal conditions.
    """
    
    # select sessions
    if dataIndex is not None:
        mode = "single"
    else:
        mode = "average"
    

    #fig, AX = pt.figure(axes_extents=[[ [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1]]])  #generalize 

    fig, AX = pt.figure(axes_extents=[[[1,1]] * 8,   # row 0: contrast 1 - generalize
                                      [[1,1]] * 8],  # row 1: contrast 2 - generalize
                        top=2, 
                        bottom = 2,
                        right = 2, 
                        left = 2, 
                        figsize=(10,2))
                        #ax_scale=(2, 11))   
    
    session_traces = []

    for data in data_s:
        episodes = EpisodeData(data,
                               quantities=['dFoF', 'Running-Speed'],
                               protocol_name=protocol,
                               prestim_duration=1,
                               verbose=False)

        if metric is not None:
            cond = compute_high_arousal_cond(episodes, pupil_threshold, running_speed_threshold, metric=metric)
        else:
            cond = episodes.find_episode_cond()
        
        varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
        varied_values = [episodes.varied_parameters[k] for k in varied_keys]


        orientations = episodes.varied_parameters['angle'] 
        contrasts    = episodes.varied_parameters['contrast'] 

        for c_idx, contrast in enumerate(contrasts):
            for o_idx, orientation in enumerate(orientations):

                stim_cond = episodes.find_episode_cond(key=['angle', 'contrast'],
                                                       value=[orientation, contrast])

                mean_trace, sem_trace = get_trial_average_trace(episodes,
                                                                roiIndex=roiIndex,
                                                                condition=stim_cond & cond)

                if mean_trace is not None:
                    session_traces.append((c_idx, o_idx, mean_trace, sem_trace))

    # plotting
    for c_idx in range(len(contrasts)):
        for o_idx in range(len(orientations)):
            
            traces = [tr for c, o, tr, _ in session_traces if c == c_idx and o == o_idx]
            sems   = [se for c, o, _, se in session_traces if c == c_idx and o == o_idx]

            if not traces:
                continue

            if mode == "single":
                mean_trace = traces[0]
                sem_trace  = sems[0]
            else:
                mean_trace = np.mean(traces, axis=0)
                sem_trace  = np.std(traces, axis=0) / np.sqrt(len(traces))

            if data.protocols[0]=="ff-gratings-2orientations-8contrasts-15repeats":
                ax = AX[o_idx][c_idx]
            elif data.protocols[0]=="ff-gratings-8orientation-2contrasts-15repeats":
                ax = AX[c_idx][o_idx]
            
            ax.plot(episodes.t, mean_trace, color='k')
            time_max = episodes.time_duration[0] + 1 #assumaes prestim 1

            ylim_enhancement=.8
            ymin, ymax = ax.get_ylim()
            dy = ymax-ymin
            ylim = [ymin-ylim_enhancement*dy/100.,ymax+ylim_enhancement*dy/100.]
            print("ylim : ", ylim)

            pt.set_plot(ax, 
                        spines = ['left', 'bottom'],
                        xticks=np.arange(-1, time_max+1, 1), 
                        xlabel='Time (s)',
                        xlim=[episodes.t[0], episodes.t[-1]], 
                        ylim=ylim)
        
            ax.fill_between(episodes.t,
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color='k',
                            alpha=0.3)

            ax.axvspan(0,
                       episodes.time_duration[0],
                       color='lightgrey',
                       alpha=0.5,
                       zorder=0)

    if data.protocols[0]=="ff-gratings-2orientations-8contrasts-15repeats":
        AX[0][0].set_ylabel("a = 0  \n dFoF")
        AX[1][0].set_ylabel("a = 90 \n dFoF")
        # Label columns
        for c_idx, contrast in enumerate(contrasts):
            AX[1][c_idx].set_xlabel(f"Time (s) \n c = {contrast:.2f}")

    elif data.protocols[0]=="ff-gratings-8orientation-2contrasts-15repeats":
        #label rows
        AX[0][0].set_ylabel("C = 0.5 \n dFoF")
        AX[1][0].set_ylabel("C = 1 \n dFoF")
        # Label columns
        for o_idx, orientation in enumerate(orientations):
            AX[1][o_idx].set_xlabel(f"Time (s) \n a = {orientation:.1f}°")

    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            AX[1][-1].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (-3, -1.5), xycoords='axes fraction')
        else:
            AX[1][-1].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (-3, -1.5), xycoords='axes fraction')
    else:
        if mode == "single":
            AX[1][-1].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data_s[0].filename.replace('.nwb','')),
                               (-3, -1.5), xycoords='axes fraction', fontsize=7)
        else:
            AX[1][-1].annotate('roi #%i , average over %i sessions' % (1+roiIndex, len(data_s)),
                               (-3, -1.5), xycoords='axes fraction', fontsize=7)

    return fig, AX

def plot_dFoF_per_protocol2(data_s,
                           dataIndex=None,
                           roiIndex=None,
                           pupil_threshold=2.9,
                           running_speed_threshold=0.1, 
                           metric=None, 
                           found=True):
    """
    Plot dFoF per protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    roiIndex : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : str or None
        Metric to split high/low arousal conditions.
    """
    # select sessions
    if dataIndex is not None:
        mode = "single"
    else:
        mode = "average"
    
    # protocols (assume same across sessions)
    protocols = [p for p in data_s[0].protocols 
                 if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]
    


    fig, AX = pt.figure(axes = (len(protocols),1))

    for p, protocol in enumerate(protocols):
        session_traces = []

        for data in data_s:
            episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Running-Speed'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)

            if metric is not None:
                cond = compute_high_arousal_cond(episodes, pupil_threshold, running_speed_threshold, metric=metric)
            else:
                cond = episodes.find_episode_cond()
            

            # TO FIX : find a better solution
            varied_keys = [k for k in episodes.varied_parameters.keys() if (k != 'repeat') and (k != 'angle') and (k != 'contrast') and (k != 'speed') and (k != 'Image-ID') and (k != 'seed')]
            varied_values = [episodes.varied_parameters[k] for k in varied_keys]


            i = 0
            for values in itertools.product(*varied_values):
                stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

                mean_trace, sem_trace = get_trial_average_trace(
                    episodes,
                    roiIndex=roiIndex,
                    condition=stim_cond & cond
                )
                if mean_trace is not None:
                    session_traces.append((i, mean_trace, sem_trace))
                i += 1

        # plotting
        n_conditions = len(protocols)
        for j in range(n_conditions):
            traces = [tr for idx, tr, _ in session_traces if idx == j]
            sems   = [se for idx, _, se in session_traces if idx == j]

            if len(traces) == 0:
                continue  # nothing to plot for this condition

            if mode == "single":
                mean_trace = traces[0]
                sem_trace  = sems[0]
            else:
                mean_trace = np.mean(traces, axis=0)
                sem_trace  = np.std(traces, axis=0) / np.sqrt(len(traces))

            AX[p].plot(mean_trace, color='k', linewidth=0.1)
            AX[p].fill_between(np.arange(len(mean_trace)),
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color='k', alpha=0.3)
            AX[p].axvspan(1000, 1000+1000*episodes.time_duration[0], color='lightgrey', alpha=0.5, zorder=0)

        AX[p].set_title(protocol.replace('Natural-Images-4-repeats','natural-images'))    
        
        #AX[p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
        #                  (0.5,1.4),
        #                  xycoords='axes fraction', ha='center', fontsize=7)
    
    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            AX[0].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (0, -0.2), xycoords='axes fraction')
            
        else:
            AX[0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (0, -0.2), xycoords='axes fraction')
            
    else:
        if mode == "single":
            AX[0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data_s[0].filename.replace('.nwb','')),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX[0].annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        else:

            AX[0].annotate('roi #%i , average over %i sessions' % (1+roiIndex, len(data_s)),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX[0].annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
    pt.set_common_xlims(AX)
    
    return fig, AX

def plot_dFoF_of_protocol2(data_s,
                           dataIndex=None,
                           roiIndex=None,
                           pupil_threshold=2.9,
                           running_speed_threshold=0.1, 
                           metric=None, 
                           found=True):
    """
    Plot dFoF of the protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    roiIndex : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : str or None
        Metric to split high/low arousal conditions.
    """
    # select sessions
    if dataIndex is not None:
        mode = "single"
    else:
        mode = "average"
    
    # protocols (assume same across sessions)
    protocol = data_s[0].protocols[0] 
    
    fig, AX = pt.figure(axes = (1,1))
    session_traces = []

    for data in data_s:
        episodes = EpisodeData(data,
                                quantities=['dFoF', 'Running-Speed'],
                                protocol_name=protocol,
                                prestim_duration=1,
                                verbose=False)

        if metric is not None:
            cond = compute_high_arousal_cond(episodes, pupil_threshold, running_speed_threshold, metric=metric)
        else:
            cond = episodes.find_episode_cond()
        

        # TO FIX : find a better solution
        varied_keys = [k for k in episodes.varied_parameters.keys() if (k != 'repeat') and (k != 'angle') and (k != 'contrast') and (k != 'speed') and (k != 'Image-ID') and (k != 'seed')]
        varied_values = [episodes.varied_parameters[k] for k in varied_keys]


        i = 0
        for values in itertools.product(*varied_values):
            stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

            mean_trace, sem_trace = get_trial_average_trace(
                episodes,
                roiIndex=roiIndex,
                condition=stim_cond & cond
            )
            if mean_trace is not None:
                session_traces.append((i, mean_trace, sem_trace))
            i += 1

    # plotting
    n_conditions = len(protocol)
    for j in range(n_conditions):
        traces = [tr for idx, tr, _ in session_traces if idx == j]
        sems   = [se for idx, _, se in session_traces if idx == j]

        if len(traces) == 0:
            continue  # nothing to plot for this condition

        if mode == "single":
            mean_trace = traces[0]
            sem_trace  = sems[0]
        else:
            mean_trace = np.mean(traces, axis=0)
            sem_trace  = np.std(traces, axis=0) / np.sqrt(len(traces))

        AX.plot(mean_trace, color='k', linewidth=0.1)
        AX.fill_between(np.arange(len(mean_trace)),
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color='k', alpha=0.3)
        AX.axvspan(1000, 1000+1000*episodes.time_duration[0], color='lightgrey', alpha=0.5, zorder=0)

    AX.set_title(protocol.replace('Natural-Images-4-repeats','natural-images'))    
    
    #AX[p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
    #                  (0.5,1.4),
    #                  xycoords='axes fraction', ha='center', fontsize=7)

    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            AX.annotate('single session: %s ,   n=%i ROIs' %
                                (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                                (0, -0.2), xycoords='axes fraction')
            
        else:
            AX.annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                                (0, -0.2), xycoords='axes fraction')
            
    else:
        if mode == "single":
            AX.annotate('roi #%i ,   rec: %s' % (1+roiIndex, data_s[0].filename.replace('.nwb','')),
                                (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX.annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        else:

            AX.annotate('roi #%i , average over %i sessions' % (1+roiIndex, len(data_s)),
                                (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX.annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        
    return fig, AX
