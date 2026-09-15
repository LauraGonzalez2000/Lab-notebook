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
from physion.analysis.episodes.trial_statistics import pre_post_statistics
from scipy import stats

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

def get_trial_average_trace(episodes,
                            quantity='dFoF',
                            index=None,
                            condition=None,
                            with_std_over_rois=False):
    """
    Return trial-averaged response trace (mean and SEM) for one Episodes object.
    """
    if condition is None:
        condition = np.ones(np.sum(episodes.protocol_cond_in_full_data), 
                            dtype=bool)
    elif len(condition) == len(episodes.protocol_cond_in_full_data):
        condition = condition[episodes.protocol_cond_in_full_data]

    avg_dim = 'episodes' if with_std_over_rois else 'ROIs'
    
    response = episodes.get_response2D(quantity=quantity,
                                       episode_cond=condition,
                                       index=index,
                                       averaging_dimension=avg_dim)
    if response.size == 0:
        return None, None

    mean_trace = response.mean(axis=0)
    sem_trace  = response.std(axis=0) / np.sqrt(response.shape[0])

    return mean_trace, sem_trace

FF_GRATINGS_2ORI = "ff-gratings-2orientations-8contrasts-15repeats"
FF_GRATINGS_8ORI = "ff-gratings-8orientation-2contrasts-15repeats"
FFSG_OPTO = "ffSG-8ori-2ctrst+1sPrePostOpto"
NATURAL_IMAGES_C = "2NaturalImages-8contrasts-15repeats"

MOVING_DOTS = "moving-dots"
DRIFTING_GRATING = "drifting-grating"
STATIC_PATCH = "static-patch"
DRIFTING_GRATINGS = "drifting-gratings"
NATURAL_IMAGES = "Natural-Images-4-repeats"

FIGURE_CONFIG = {

    FF_GRATINGS_2ORI: {
        "shape": (2, 8),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
        "wspace": 1.4,
        "hspace": 1.8,
    },

    FF_GRATINGS_8ORI: {
        "shape": (2, 8),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
        "wspace": 1.4,
        "hspace": 1.8,
    },

    NATURAL_IMAGES_C: {
        "shape": (2, 8),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
        "wspace": 1.4,
        "hspace": 1.8,
    },

    DRIFTING_GRATING: {
        "shape": (1, 3),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
    },

    STATIC_PATCH: {
        "shape": (1, 2),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
    },

    DRIFTING_GRATINGS: {
        "shape": (1, 4),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
    },

    NATURAL_IMAGES: {
        "shape": (1, 5),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.8),
    },

    FFSG_OPTO: {
        "shape": (4, 9),
        "figsize": (10, 4),
        "ax_scale": (1.2, 1.5),
        "top": 10,
    },
}

def create_protocol_fig(protocol, protocols):

    first_protocol = protocols[0]

    # Select the configuration
    if first_protocol == MOVING_DOTS:
        config = FIGURE_CONFIG[protocol]
    elif DRIFTING_GRATING in protocols:
        config = FIGURE_CONFIG[DRIFTING_GRATING]
    else:
        config = FIGURE_CONFIG[first_protocol]

    rows, cols = config["shape"]

    axes_extents = [[[1, 1]] * cols for _ in range(rows)]

    fig, AX = pt.figure(
        axes_extents=axes_extents,
        top=config.get("top", 8),
        bottom=config.get("bottom", 8),
        right=config.get("right", 2),
        left=config.get("left", 2),
        figsize=config["figsize"],
        ax_scale=config["ax_scale"],
        wspace=config.get("wspace", 1.5),
        hspace=config.get("hspace", 1.5),
    )
    return fig, AX

def get_values(data_s, index, opto, protocol, metric, pupil_threshold, running_speed_threshold):

    session_traces = []
    session_traces_control = []
    session_traces_opto = []

    for i, data in enumerate(data_s):

        if index is None : 
            print("file  ", i,", all ", data.nROIs, " rois selected")
        else: 
            print("file : ", i,", selected cells :", index," total rois :", data.nROIs)
            
        if opto: 
            episodes = EpisodeData(data,
                                quantities=['dFoF', 'running', 'opto'],
                                protocol_name=protocol,
                                prestim_duration=2, 
                                verbose=False)
            LED_on = episodes.opto.mean(axis=1)>0

        else: 
            episodes = EpisodeData(data,
                                quantities=['dFoF', 'running'],
                                protocol_name=protocol,
                                prestim_duration=1, #FIX
                                verbose=False)
        
        if metric is not None:
            cond = compute_high_arousal_cond(episodes, pupil_threshold, 
                                                running_speed_threshold, 
                                                metric=metric)
        else:
            cond = episodes.find_episode_cond()
        
        varied_keys = [k for k in episodes.varied_parameters.keys()
                        if k!='repeat']

        if len(varied_keys)==2:
            key1 = episodes.varied_parameters[varied_keys[0]] 
            key2    = episodes.varied_parameters[varied_keys[1]] 

            for key2_idx, key2_ in enumerate(key2):
                for key1_idx, key1_ in enumerate(key1):
                    stim_cond = episodes.find_episode_cond(key=[varied_keys[0], 
                                                                varied_keys[1]],
                                                        value=[key1_, key2_])
                    if opto: 
                        mean_trace_control, sem_trace_control = get_trial_average_trace(episodes,
                                                        index=index,
                                                        condition=stim_cond & cond & ~LED_on)
                        
                        mean_trace_opto, sem_trace_opto = get_trial_average_trace(episodes,
                                                        index=index,
                                                        condition=stim_cond & cond & LED_on)
                        
                        if mean_trace_control is not None:
                            session_traces_control.append((key2_idx, key1_idx, 
                                                mean_trace_control, sem_trace_control))
                        if mean_trace_opto is not None:
                            session_traces_opto.append((key2_idx, key1_idx, 
                                                mean_trace_opto, sem_trace_opto))
                            
                    else: 
                        mean_trace, sem_trace = get_trial_average_trace(episodes,
                                                        index=index,
                                                        condition=stim_cond & cond)
                        if mean_trace is not None:
                            session_traces.append((key2_idx, key1_idx, 
                                                mean_trace, sem_trace))

        elif len(varied_keys)==1:
            
            key1 = episodes.varied_parameters[varied_keys[0]] 
            
            for key1_idx, key1_ in enumerate(key1):
                stim_cond = episodes.find_episode_cond(key=varied_keys[0],
                                                        value=key1_)
                mean_trace, sem_trace = get_trial_average_trace(episodes,
                                                                index=index,
                                                                condition=stim_cond & cond)
                if mean_trace is not None:
                    session_traces.append((key1_idx, mean_trace, sem_trace))

        protocol_info = {"varied_keys": varied_keys,
                         "key1": key1,
                         "key2": key2}

    return session_traces, session_traces_control, session_traces_opto, protocol_info, data, episodes

def normalize_trace(trace, baseline_samples=1000):
    baseline = np.nanmean(trace[:baseline_samples])
    return trace - baseline

def plot_trace(ax, time, trace, sem, color, normalize=False):
    if normalize:
        trace = normalize_trace(trace)
    ax.plot(time, trace, color=color)
    ax.fill_between(time,
                    trace - sem,
                    trace + sem,
                    color=color,
                    alpha=0.3)
    return 0

def plot_(protocol_info, session_traces_control, session_traces_opto, 
          mode, opto, data, AX, norm, episodes, color_trace, session_traces, 
          data_s, subplots_n, index, protocol, ylim ):
    # plotting
    if len(protocol_info["varied_keys"])==2:
        for key2_idx in range(len(protocol_info["key2"])):
            for key1_idx in range(len(protocol_info["key1"])):
                
                if opto: 
                    traces_control = [tr for c, o, tr, se in session_traces_control 
                            if c == key2_idx and o == key1_idx]
                    sems_control   = [se for c, o, tr, se in session_traces_control 
                            if c == key2_idx and o == key1_idx]
                    
                    traces_opto = [tr for c, o, tr, se in session_traces_opto 
                            if c == key2_idx and o == key1_idx]
                    sems_opto   = [se for c, o, tr, se in session_traces_opto 
                            if c == key2_idx and o == key1_idx]

                    if (not traces_control) or (not traces_opto):
                        continue

                    if mode == "single":
                        mean_trace_control = traces_control[0]
                        sem_trace_control  = sems_control[0]

                        mean_trace_opto = traces_opto[0]
                        sem_trace_opto  = sems_opto[0]
                    else:
                        mean_trace_control = np.nanmean(traces_control, axis=0)
                        sem_trace_control  = np.nanstd(traces_control, axis=0) / np.sqrt(len(traces_control))
                        
                        mean_trace_opto = np.nanmean(traces_opto, axis=0)
                        sem_trace_opto  = np.nanstd(traces_opto, axis=0) / np.sqrt(len(traces_opto))

                    if data.protocols[0]=="ffSG-8ori-2ctrst+1sPrePostOpto":
                        ax_control = AX[2*key2_idx][key1_idx]
                        ax_opto  = AX[2*key2_idx+1][key1_idx]

                    if norm :
                        mean_trace_control = normalize_trace(mean_trace_control)
                        mean_trace_opto = normalize_trace(mean_trace_opto)

                    plot_trace(ax_control,
                               episodes.t,
                               mean_trace_control,
                               sem_trace_control,
                               color_trace)
                        
                    if opto: 
                        plot_trace(ax_opto,
                                   episodes.t,
                                   mean_trace_opto,
                                   sem_trace_opto,
                                   color_trace)
                        ax_opto.axvspan(-1, 3, color='navy',alpha=0.2, lw=0, zorder=-10)

                    time_max = episodes.time_duration[0] + 1 #assumes prestim 1

                    ymin, ymax = ax_control.get_ylim()
                    dy = ymax-ymin
                    ylim = ylim  #[ymin-0.7*dy,ymax+0.7*dy]
                    #ylim_fixed = ylim

                    pt.set_plot(ax_control, 
                                spines = ['left', 'bottom'],
                                xticks=np.arange(-2, time_max+2, 1), 
                                xlabel='Time (s)',
                                xlim=[episodes.t[0], episodes.t[-1]], 
                                ylim=ylim)
                
                    ax_control.axvspan(0,
                            episodes.time_duration[0],
                            color='lightgrey',
                            alpha=0.5,
                            zorder=0)
                    


                    ymin, ymax = ax_opto.get_ylim()
                    dy = ymax-ymin
                    ylim = ylim  #[ymin-0.7*dy,ymax+0.7*dy]
                    #ylim_fixed = ylim

                    pt.set_plot(ax_opto, 
                                spines = ['left', 'bottom'],
                                xticks=np.arange(-2, time_max+2, 1), 
                                xlabel='Time (s)',
                                xlim=[episodes.t[0], episodes.t[-1]], 
                                ylim=ylim)
                
                    ax_opto.axvspan(0,
                            episodes.time_duration[0],
                            color='lightgrey',
                            alpha=0.5,
                            zorder=0)

                    
                else: 
                    traces = [tr for c, o, tr, se in session_traces 
                            if c == key2_idx and o == key1_idx]
                    sems   = [se for c, o, tr, se in session_traces 
                            if c == key2_idx and o == key1_idx]

                    if not traces:
                        continue

                    if mode == "single":
                        mean_trace = traces[0]
                        sem_trace  = sems[0]
                    else:
                        mean_trace = np.nanmean(traces, axis=0)
                        sem_trace  = np.nanstd(traces, axis=0) / np.sqrt(len(traces))

                    if data.protocols[0]=="ff-gratings-2orientations-8contrasts-15repeats" or \
                        data.protocols[0]=="2NaturalImages-8contrasts-15repeats":
                        ax = AX[key1_idx][key2_idx]
                    elif data.protocols[0]=="ff-gratings-8orientation-2contrasts-15repeats":
                        ax = AX[key2_idx][key1_idx]
                    elif data.protocols[0]=="ffSG-8ori-2ctrst+1sPrePostOpto":
                        ax = AX[key2_idx][key1_idx]

                    
                    if norm : 
                        mean_trace = normalize_trace(mean_trace)
                    
                    plot_trace(ax,
                               episodes.t,
                               mean_trace,
                               sem_trace,
                               color_trace)


                    time_max = episodes.time_duration[0] + 1 #assumes prestim 1

                    ymin, ymax = ax.get_ylim()
                    dy = ymax-ymin
                    ylim = ylim  #[ymin-0.7*dy,ymax+0.7*dy]
                    #ylim_fixed = ylim

                    pt.set_plot(ax, 
                                spines = ['left', 'bottom'],
                                xticks=np.arange(-1, time_max+1, 1), 
                                xlabel='Time (s)',
                                xlim=[episodes.t[0], episodes.t[-1]], 
                                ylim=ylim, 
                                fontsize=15)
                
                    ax.axvspan(0,
                            episodes.time_duration[0],
                            color='lightgrey',
                            alpha=0.5,
                            zorder=0)
        
        plot_barplot2_of_protocol(data_s = data_s, 
                                    AX = AX[1][8], 
                                    idx = 0, 
                                    p=["ffSG-8ori-2ctrst+1sPrePostOpto"], 
                                    subplots_n= subplots_n, 
                                    subset_rois = index,
                                    stat_test_props={}, 
                                    color_bar=color_trace,
                                    values = [1, 2, 1, 6, 3, 4, 8, 9],
                                yerr = [1, 2, 1, 2, 3, 3, 2, 1],
                                param_values=["a=0", "a=22.5", "a=45", "a=67.5",
                                                "a=90","a=112.5","a=135","a=157.5"], 
                                    opto=True)
        plot_barplot2_of_protocol(data_s = data_s, 
                                    AX = AX[2][8], 
                                    idx = 0, 
                                    p=["ffSG-8ori-2ctrst+1sPrePostOpto"], 
                                    subplots_n= subplots_n, 
                                    subset_rois = index,
                                    stat_test_props={},
                                values = [1, 2, 1, 6, 3, 4, 8, 9],
                                yerr = [1, 2, 1, 2, 3, 3, 2, 1],
                                param_values=["a=0", "a=22.5", "a=45", "a=67.5",
                                                "a=90","a=112.5","a=135","a=157.5"],  
                                    color_bar=color_trace)
        plot_barplot2_of_protocol(data_s = data_s, 
                                    AX = AX[3][8], 
                                    idx = 0, 
                                    p=["ffSG-8ori-2ctrst+1sPrePostOpto"], 
                                    subplots_n= subplots_n, 
                                    subset_rois = index,
                                    stat_test_props={}, 
                                    color_bar=color_trace, 
                                    values = [1, 2, 1, 6, 3, 4, 8, 9],
                                yerr = [1, 2, 1, 2, 3, 3, 2, 1],
                                param_values=["a=0", "a=22.5", "a=45", "a=67.5",
                                                "a=90","a=112.5","a=135","a=157.5"], 
                                    opto=True)
                
        

        if data.protocols[0]=="ff-gratings-2orientations-8contrasts-15repeats":
            if norm : 
                AX[0][0].set_ylabel("a = 0  \n dFoF\n (normalized)")
                AX[1][0].set_ylabel("a = 90 \n dFoF\n (normalized)")
            else: 
                AX[0][0].set_ylabel("a = 0  \n dFoF ")
                AX[1][0].set_ylabel("a = 90 \n dFoF ")
            # Label columns
            for c_idx, contrast in enumerate(protocol_info["key2"]):
                AX[1][c_idx].set_xlabel(f"Time (s) \n c = {contrast:.2f}")

        elif data.protocols[0]=="ff-gratings-8orientation-2contrasts-15repeats":
            #label rows
            if norm: 
                AX[0][0].set_ylabel(" C = 0.5 \ndFoF (normalized)")
                AX[1][0].set_ylabel(" C = 1 \ndFoF (normalized)")
            else: 
                AX[0][0].set_ylabel(" C = 0.5 \ndFoF ")
                AX[1][0].set_ylabel(" C = 1 \ndFoF ")
            # Label columns
            for o_idx, orientation in enumerate(protocol_info["key1"]):
                AX[1][o_idx].set_xlabel(f"Time (s) \n\n a = {orientation:.1f}°")
        
        elif data.protocols[0]=="ffSG-8ori-2ctrst+1sPrePostOpto":
            #label rows
            if norm: 
                AX[0][0].set_ylabel(" C = 0.5 \ndFoF (normalized)")
                AX[1][0].set_ylabel("OPTO\n C = 0.5 \ndFoF (normalized)")
                AX[2][0].set_ylabel(" C = 1 \ndFoF (normalized)")
                AX[3][0].set_ylabel("OPTO\n C = 1 \ndFoF (normalized)")
            else: 
                AX[0][0].set_ylabel(" C = 0.5 \ndFoF ")
                AX[1][0].set_ylabel("OPTO\n C = 0.5 \ndFoF")
                AX[2][0].set_ylabel(" C = 1 \ndFoF ")
                AX[3][0].set_ylabel("OPTO\n C = 1 \ndFoF")
            # Label columns
            for o_idx, orientation in enumerate(protocol_info["key1"]):
                AX[3][o_idx].set_xlabel(f"Time (s) \n\n a = {orientation:.1f}°")


        # annotate session or ROI info
        if index is None :
            if mode == "single":
                AX[-1][-1].annotate('single session: %s ,   n=%i ROIs' %
                                (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                                (-2, -1), xycoords='axes fraction')
            else:
                AX[-1][-1].annotate('average over %i sessions ,   ' \
                'mean$\\pm$SEM across sessions' % len(data_s),
                                (-4, -1.5), xycoords='axes fraction', 
                                fontsize=15)
        else :
            if isinstance(index, (int, np.integer)):
                if mode == "single":
                    AX[-1][-1].annotate('roi #%i ,   rec: %s' 
                                    % (1+len(index), data_s[0].filename.replace('.nwb','')),
                                    (-2, -1), xycoords='axes fraction', fontsize=7)
                else:
                    AX[-1][-1].annotate('roi #%i , average over %i sessions' 
                                    % (1+len(index), len(data_s)),
                                    (-2, -1), xycoords='axes fraction', fontsize=7)
            elif isinstance(index, list):
                if len(index)>1:
                    if mode == "single":
                        AX[-1][-1].annotate('roi subset ,   rec: %s' 
                                            % ( data_s[0].filename.replace('.nwb','')),
                                        (-2, -1), xycoords='axes fraction', fontsize=7)
                    else:
                        AX[-1][-1].annotate('roi subset , average over %i sessions' 
                                            % ( len(data_s)),
                                        (-2, -1), xycoords='axes fraction', fontsize=7)
    
    elif len(protocol_info["varied_keys"])==1:
        for key1_idx in range(len(protocol_info["key1"])):
                
                traces = [tr for c, tr, se in session_traces if c == key1_idx ]
                sems   = [se for c, tr, se in session_traces if c == key1_idx ]

                if not traces:
                    continue

                if mode == "single":
                    mean_trace = traces[0]
                    sem_trace  = sems[0]
                else:
                    mean_trace = np.nanmean(traces, axis=0)
                    sem_trace  = np.nanstd(traces, axis=0) / np.sqrt(len(traces))
                ax = AX[key1_idx]

                if norm : 
                    mean_trace = normalize_trace(mean_trace)
                                    
                plot_trace(ax,
                           episodes.t,
                           mean_trace,
                           sem_trace,
                           color_trace)
                
                AX[0].set_ylabel("dFoF (normalized)" if norm else "dFoF")

                        
                time_max = episodes.time_duration[0] + 1 #assumes prestim 1

                ymin, ymax = ax.get_ylim()
                dy = ymax-ymin
                ylim = ylim #[ymin-0.7*dy,ymax+0.7*dy]

                pt.set_plot(ax, 
                            spines = ['left', 'bottom'],
                            xticks=np.arange(-1, time_max+1, 1), 
                            xlabel='Time (s)',
                            xlim=[episodes.t[0], episodes.t[-1]], 
                            ylim=ylim)
            
                ax.axvspan(0,
                        episodes.time_duration[0],
                        color='lightgrey',
                        alpha=0.5,
                        zorder=0)

        
        
        if data.protocols[0]=="ff-gratings-8orientation-2contrasts-15repeats":
            for c_idx, contrast in enumerate(protocol_info["key1"]):
                AX[c_idx].set_xlabel(f"Time (s) \n c = {contrast:.2f}")
        if protocol == "static-patch":
            for c_idx, contrast in enumerate(protocol_info["key1"]):
                AX[c_idx].set_xlabel(f"Time (s) \n a = {contrast:.2f}")
        if protocol == "drifting-gratings":
            for c_idx, contrast in enumerate(protocol_info["key1"]):
                AX[c_idx].set_xlabel(f"Time (s) \n dir = {contrast}")
        if protocol == "Natural-Images-4-repeats":
            for c_idx, contrast in enumerate(protocol_info["key1"]):
                AX[c_idx].set_xlabel(f"Time (s) \n ID = {contrast}")


            
        # annotate session or ROI info
        if index is None:
            if mode == "single":
                AX[-1].annotate('single session: %s ,   n=%i ROIs' %
                                (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                                (-2, -1), xycoords='axes fraction')
            else:
                AX[-1].annotate('average over %i sessions ,  ' \
                ' mean$\\pm$SEM across sessions' % len(data_s),
                                (-2, -1), xycoords='axes fraction')
        else:
            if index.type == np.float: 
                if mode == "single":
                    AX[-1].annotate('roi #%i ,   rec: %s' 
                                    % (1+len(index), data_s[0].filename.replace('.nwb','')),
                                    (-2, -1), xycoords='axes fraction', fontsize=7)
                else:
                    AX[-1].annotate('roi #%i , average over %i sessions' 
                                    % (1+len(index), len(data_s)),
                                    (-2, -1), xycoords='axes fraction', fontsize=7)
                    
            elif index.type == np.list :
                if mode == "single":
                    AX[-1].annotate('roi subset ,   rec: %s' 
                                    % ( data_s[0].filename.replace('.nwb','')),
                                    (-2, -1), xycoords='axes fraction', fontsize=7)
                else:
                    AX[-1].annotate('roi subset , average over %i sessions' 
                                    % ( len(data_s)),
                                    (-2, -1), xycoords='axes fraction', fontsize=7)
    return 0

def plot_dFoF_of_protocol(data_s,
        dataIndex=None,
        index=None,
        pupil_threshold=2.9,
        running_speed_threshold=0.5, 
        metric=None, 
        protocol = "", 
        ylim = [-2, 0.15],
        norm=True,
        opto=False,
        color_trace = 'k', 
        subplots_n=16, 
        quantification= False):
    
    """
    Plot dFoF per protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    index : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : "pupil" or "locomotion"
        Metric from which to split high/low arousal conditions.
    protocol : string
        protocol of interest ( is first protocol useful? Or we shift argument to be mandatory, anyways it's always only one protocol)
    ylim : list []
        defines ylim of plots
    norm : Boolean
        if True, traces are normalized. Default to True
    opto : Boolean
        if True, optogenetics is being used. (Double the graphs)
    color_trace : string
        color of trace
    subplots_n : int
        number of barplots (to change or put elsewhere!)

    quantification : Boolean
        If True, barplots are added. 
    
    --------

    Returns : fig, AX
     
    """

    # select sessions
    
    if dataIndex is not None:
        mode = "single"
        data_s = [data_s[dataIndex]]
    else:
        mode = "average"

    fig, AX = create_protocol_fig( protocol, protocols = data_s[0].protocols )

    session_traces, session_traces_control, session_traces_opto, protocol_info, data, episodes = \
        get_values(data_s, index, opto, protocol, metric, pupil_threshold, running_speed_threshold)

    plot_(protocol_info, session_traces_control, session_traces_opto, 
          mode, opto, data, AX, norm, episodes, color_trace, session_traces, 
          data_s, subplots_n, index, protocol, ylim )

    return fig, AX

def plot_barplot2_of_protocol(data_s, AX, idx,  p, 
                              subplots_n, subset_rois=None, 
                              stat_test_props={}, color_bar='k', 
                              values = None,
                              param_values = None,
                              yerr = None,
                              opto=False):
    """
    Takes as arguments : 
        data_s -> 
        AX -> 
        idx ->
        p -> 
        subplots_n -> 
        subset_rois -> 
        stat_test_props -> 

    Method:
        calculates the summary data per file. 
        The values are stored in mean_vals (size #num_values_per_file) 
        and concatenated in mean_vals_s (size #files x #num_values_per_file)
        The mean is calculated (average all files) (size #num_values_per_file)
        The sem is calculated (similarity between files) (size #num_values_per_file)

    Output:
    The barplot is plotted

    """
    if values == None : #calculate variation traces
        mean_vals_s = []  # store per-session mean responses

        for data_i, data in enumerate(data_s):
            if opto :
                ep = EpisodeData(data, protocol_name=p[0], quantities=['dFoF', 'opto'])
            else: 
                ep = EpisodeData(data, protocol_name=p[0], quantities=['dFoF'])
            t0 = max([0, ep.time_duration[0] - 1.0])

            if stat_test_props=={}:
                stat_test_props = dict(
                    interval_pre=[-1.0, 0],
                    interval_post=[t0, t0 + 1.0],
                    test='ttest',
                    sign='both')
            
            if subset_rois == None: 
                if opto: 
                    LED_on = ep.opto.mean(axis=1)>0
                    summary_data = pre_post_statistics(ep,
                                                    episode_cond = LED_on,
                                                    response_args = {},
                                                    response_significance_threshold=0.05,
                                                    stat_test_props=stat_test_props,
                                                    repetition_keys=['repeat'])
                
                else: 
                    summary_data = pre_post_statistics(ep,
                                                    episode_cond = ep.find_episode_cond(),
                                                    response_args = {},
                                                    response_significance_threshold=0.05,
                                                    stat_test_props=stat_test_props,
                                                    repetition_keys=['repeat'])
            
                # Extract ROI mean values
                #mean_vals = np.nanmean(vals_subset, axis=0) #easier no?
                mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

            else :
                summary_data = pre_post_statistics(ep,
                                            episode_cond = ep.find_episode_cond(),
                                            response_args = {'quantity': "dFoF"},
                                            response_significance_threshold=0.05,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=['repeat'], 
                                            loop_over_cells=True) # Loop over all cells!!
                
                subset_rois_i = subset_rois[data_i]
                vals_subset = summary_data['value'][subset_rois_i]
                mean_vals = np.nanmean(vals_subset, axis=0)
                #mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

            # Pad/truncate to subplots_n elements - necessary??
            target_len = subplots_n
            mean_vals = (list(mean_vals) + [np.nan] * target_len)[:target_len]

            mean_vals_s.append(mean_vals)
        
        ep0 = EpisodeData(data_s[0], protocol_name=p[0], quantities=['dFoF'])

        varied_keys = list(ep0.varied_parameters.keys())
        angles = ep0.varied_parameters[varied_keys[0]]
        contrasts = ep0.varied_parameters[varied_keys[1]]

        if p[0] == 'ff-gratings-8orientation-2contrasts-15repeats':
            param_values = [f"a={a:.1f}° , C={c:.1f}" for c in contrasts for a in angles]
        elif p[0] == 'ffSG-8ori-2ctrst+1sPrePostOpto':
            param_values = [f"a={a:.1f}° , C={c:.1f}" for c in contrasts for a in angles]
        elif p[0] == 'ff-gratings-2orientations-8contrasts-15repeats':
            param_values = [f"a={a:.1f}° , C={c:.2f}" for a in angles for c in contrasts]
        elif p[0]== '2NaturalImages-8contrasts-15repeats':
            param_values = [f"Img={a:.1f} , C={c:.2f}" for a in angles for c in contrasts]
        else: 
            contrasts = ep0.varied_parameters[varied_keys[0]]
            param_values = [f"C={c:.2f}" for c in contrasts]

        # Compute session-aggregated mean and SEM
        values = np.nanmean(mean_vals_s, axis=0)
        yerr = stats.sem(mean_vals_s, axis=0, nan_policy='omit')

        # Reorder only for the 8 orientations × 2 contrasts protocol
        if p[0] == 'ff-gratings-8orientation-2contrasts-15repeats' or \
        p[0] == 'ffSG-8ori-2ctrst+1sPrePostOpto':
            n_ori = len(angles)
            n_con = len(contrasts)

            values = np.asarray(values).reshape(n_ori, n_con).T.reshape(-1)
            yerr = np.asarray(yerr).reshape(n_ori, n_con).T.reshape(-1)

    x = np.arange(len(values))

    # Plot
    AX.bar(x, values, 
           yerr=yerr,
           alpha=0.8, 
           capsize=0,
           error_kw=dict(linewidth=0.6), 
           color= color_bar)
    #pt.set_plot(ax = AX, 
    #            )
    AX.set_xticks(x)
    AX.set_xticklabels(param_values,rotation=90, ha="center")
    AX.axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX.set_ylabel('variation \ndFoF')
    
    return 0

def plot_dFoF_per_protocol(data_s,
                           dataIndex=None,
                           index=None,
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
    index : int or None
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
    print("rows", len(AX))        
    print("columns", len(AX[0]))   

    for p, protocol in enumerate(protocols):
        session_traces = []

        for data in data_s:
            episodes = EpisodeData(data,
                                   quantities=['dFoF', 'running'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)

            if metric is not None:
                cond = compute_high_arousal_cond(episodes, 
                                                 pupil_threshold, 
                                                 running_speed_threshold, 
                                                 metric=metric)
            else:
                cond = episodes.find_episode_cond()
            
            varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
            varied_values = [episodes.varied_parameters[k] for k in varied_keys]

            print(varied_keys)
            print(varied_values)

            i = 0
            for values in itertools.product(*varied_values):
                stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

                mean_trace, sem_trace = get_trial_average_trace(
                    episodes,
                    index=index,
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
    if index is None:
        if mode == "single":
            AX[-1][0].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (0, -0.2), xycoords='axes fraction')
        else:
            AX[-1][0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (0, -0.2), xycoords='axes fraction')
    else:
        if mode == "single":
            AX[-1][0].annotate('roi #%i ,   rec: %s' % (1+index, data_s[0].filename.replace('.nwb','')),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
        else:
            AX[-1][0].annotate('roi #%i , average over %i sessions' % (1+index, len(data_s)),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
    pt.set_common_xlims(AX)
    
    return fig, AX

def plot_dFoF_per_protocol2(data_s,
                           dataIndex=None,
                           index=None,
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
    index : int or None
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
                                   quantities=['dFoF', 'running'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)

            if metric is not None:
                cond = compute_high_arousal_cond(episodes, 
                                                 pupil_threshold, 
                                                 running_speed_threshold, 
                                                 metric=metric)
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
                    index=index,
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
    if index is None:
        if mode == "single":
            AX[0].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (0, -0.2), xycoords='axes fraction')
            
        else:
            AX[0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (0, -0.2), xycoords='axes fraction')
            
    else:
        if mode == "single":
            AX[0].annotate('roi #%i ,   rec: %s' % (1+index, data_s[0].filename.replace('.nwb','')),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX[0].annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        else:

            AX[0].annotate('roi #%i , average over %i sessions' % (1+index, len(data_s)),
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
                           index=None,
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
    index : int or None
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
                                quantities=['dFoF', 'running'],
                                protocol_name=protocol,
                                prestim_duration=1,
                                verbose=False)

        if metric is not None:
            cond = compute_high_arousal_cond(episodes, 
                                             pupil_threshold, 
                                             running_speed_threshold, 
                                             metric=metric)
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
                index=index,
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
    if index is None:
        if mode == "single":
            AX.annotate('single session: %s ,   n=%i ROIs' %
                                (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                                (0, -0.2), xycoords='axes fraction')
            
        else:
            AX.annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                                (0, -0.2), xycoords='axes fraction')
            
    else:
        if mode == "single":
            AX.annotate('roi #%i ,   rec: %s' % (1+index, data_s[0].filename.replace('.nwb','')),
                                (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX.annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        else:

            AX.annotate('roi #%i , average over %i sessions' % (1+index, len(data_s)),
                                (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX.annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        
    return fig, AX
