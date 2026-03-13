# %% [markdown]
# FUNCTONS USEFUL TO PLOT RESPONSIVENESS

# %% PACKAGES
import sys
import numpy as np

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.utils import plot_tools as pt
from physion.analysis.episodes.trial_statistics import pre_post_statistics
from physion.analysis.episodes.build import EpisodeData

from scipy import stats

#%% FUNCTIONS

# COMPUTES RESPONSIVENESS

def calc_responsiveness(ep, nROIs, alpha=0.05, t_window = 1.5):

    '''
    Calculates the responsiveness of each ROI in an episode
    
    takes as arguments : 
        ep -> episode
        nROIs -> number of ROIs
    
    returns : 
        3 lists of booleans (each of len nROIs)
            Responsive ROIs (True if resposive, False if not)
            Positively responsive ROIs (True if positively responsive, False if not)
            Negatively responsive ROIs (True if negatively responsive, False if not)
    '''

    session_summary = {'significant':[], 'value':[]}

    t0 = max([0, ep.time_duration[0]-t_window])
    stat_test_props = dict(interval_pre=[-t_window,0],                                   
                           interval_post=[t0, t0+t_window],                                   
                           test='ttest', 
                           sign='both')

    for roi_n in range(nROIs):
        roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=alpha,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=['repeat'])

        session_summary['significant'].append(bool(roi_summary_data['significant'][0]))
        session_summary['value'].append(roi_summary_data['value'][0])

    resp_cond = np.array(session_summary['significant'])                  
    pos_cond = resp_cond & ([session_summary['value'][i]>0 for i in range(len(session_summary['value']))])
    neg_cond = resp_cond & ([session_summary['value'][i]<0 for i in range(len(session_summary['value']))])

    print(f"{sum(resp_cond)} significant ROI \n ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) \n out of {len(session_summary['significant'])} ROIs")

    return resp_cond, pos_cond, neg_cond

def calc_responsiveness2(ep, nROIs, alpha=0.05, t_window = 1.5, repetition_keys = ['repeat']):
    
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(nROIs):
        t0 = max([0, ep.time_duration[0]-t_window])
        stat_test_props = dict(interval_pre=[-t_window,0],                                   
                                interval_post=[t0, t0+t_window],                                   
                                test='ttest', 
                                sign='both')
        
        roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=alpha,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=repetition_keys)
        
        session_summary['significant'].append(roi_summary_data['significant'])
        session_summary['value'].append(roi_summary_data['value'])

    resp_cond = np.array(session_summary['significant'])
    pos_cond = resp_cond & np.array([session_summary_value >0 for session_summary_value in session_summary['value'] ])
    neg_cond = resp_cond & np.array([session_summary_value <0 for session_summary_value in session_summary['value'] ])

    #for i in range(len(resp_cond)):
    #    print(f"{sum(resp_cond[i])} significant ROI \n ({np.sum(pos_cond[i])} positive, {np.sum(neg_cond[i])} negative) \n out of {len(session_summary['significant'])} ROIs")

    return resp_cond, pos_cond, neg_cond

def compute_responsiveness(ep, nROIs, alpha=0.05, window=1.5):
    '''
    Calculates the responsiveness of each ROI in an episode
    
    takes as arguments : 
        ep -> episode
        nROIs -> number of ROIs
    optional: 
        alpha -> response_significance_threshold (default 0.05)
        window -> time window to take values for the post and pre groups, to then make a statistical test to see if responsive
                  (default 1.5)
    
    returns : 
        1 dictionary: 
            pos_frac -> fraction positive ROIs
            neg_frac -> fraction negative ROIs
            ns_frac -> fraction ns ROIs
            n_pos -> # positive ROIS
            n_neg -> # negative ROIS
            n_sig -> # significant ROIS
            nROIs -> # total ROIs
    '''

    session_summary = {'significant': [], 'value': []}

    t0 = max([0, ep.time_duration[0] - window])

    stat_test_props = dict(
        interval_pre=[-window, 0],
        interval_post=[t0, t0 + window],
        test='ttest',
        sign='both'
    )

    for roi_n in range(nROIs):
        roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=alpha,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=list(ep.varied_parameters.keys()))
        
        session_summary['significant'].append(bool(roi_summary_data['significant']))
        session_summary['value'].append(roi_summary_data['value'])

    significant = np.array(session_summary['significant'])
    values = np.array(session_summary['value'])

    pos = significant & (values > 0)
    neg = significant & (values < 0)
    ns  = ~significant

    return dict(pos_frac=np.sum(pos) / nROIs,
                neg_frac=np.sum(neg) / nROIs,
                ns_frac=np.sum(ns) / nROIs,
                n_pos=np.sum(pos),
                n_neg=np.sum(neg),
                n_sig=np.sum(significant),
                nROIs=nROIs)

#PLOTS RESPONSIVENESS

def plot_protocol_responsiveness(ep, nROIs, AX, protocol="", idx=None, colors=['red', 'blue', 'grey']):

    '''
    Plots pie chart of responsiveness in an episode / group of episodes? 
    
    takes as arguments : 
        ep -> episode
        nROIs -> number of ROIs
        AX -> to plot
    optional: 
        protocol -> to put titles at the right place
        idx -> in case there are subplots
        colors -> colors of positive, negative and non-responsive (in that order)
    
    returns : 
        nothing, but plots the figure 
    '''

    ax = AX[idx] if idx is not None else AX

    resp = compute_responsiveness(ep, nROIs)

    print(f"Protocol {protocol} : "
          f"{resp['n_sig']} significant ROI "
          f"({resp['n_pos']} positive, {resp['n_neg']} negative) "
          f"out of {nROIs} ROIs")

    pt.pie(data=[resp['pos_frac'], resp['neg_frac'], resp['ns_frac']],
           ax=ax,
           COLORS=colors)

    protocol = protocol.replace('Natural-Images-4-repeats', 'natural-images')
    protocol = protocol.replace('ff-gratings-2orientations-8contrasts-15repeats', '2orientations-8contrasts')
    protocol = protocol.replace('ff-gratings-8orientation-2contrasts-15repeats',  '8orientations-2contrasts')

    ax.set_title(protocol)
    pt.annotate(ax, f"+ resp={100*resp['pos_frac']:.1f}%", (1, 0), ha='right', va='top')
    pt.annotate(ax, f"- resp={100*resp['neg_frac']:.1f}%", (1, -0.2), ha='right', va='top')
    pt.annotate(ax, f"{resp['nROIs']} ROIs", (1, -0.4), ha='right', va='top')

    return 0

def plot_responsiveness2_per_protocol(data_s, AX, idx, p, t_window=1.5, alpha=0.05, type='means', colors= ['red', 'blue', 'grey']):
    
    pos_s = []
    neg_s = []

    resp_cond_s = []
    pos_cond_s = []
    neg_cond_s = []

    nROIs = []

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
        
        sig_list = []
        val_list = []

        t0 = max([0, ep.time_duration[0]-t_window])
        stat_test_props = dict(interval_pre=[-t_window,0],
                               interval_post=[t0, t0+t_window],
                               test='ttest',
                               sign='both')
        
        for roi_n in range(data.nROIs):
            roi_summary_data = pre_post_statistics(ep,
                                                   episode_cond = ep.find_episode_cond(),
                                                   response_args = dict(roiIndex=roi_n),
                                                   response_significance_threshold=alpha,
                                                   stat_test_props=stat_test_props,
                                                   repetition_keys=list(ep.varied_parameters.keys()))

            sig_list.append(bool(roi_summary_data['significant'][0]))
            val_list.append(roi_summary_data['value'][0])
        nROIs.append(data.nROIs)

        sig_arr = np.array(sig_list)
        val_arr = np.array(val_list)

        resp_cond = sig_arr
        pos_cond = sig_arr & (val_arr > 0)
        neg_cond = sig_arr & (val_arr < 0)

        resp_cond_s.append(resp_cond)
        pos_cond_s.append(pos_cond)
        neg_cond_s.append(neg_cond)

        pos = np.sum(pos_cond) / len(sig_arr)
        neg = np.sum(neg_cond) / len(sig_arr)

        pos_s.append(pos)
        neg_s.append(neg)

    if type== 'means':
        final_pos = np.mean(pos_s)
        final_neg = np.mean(neg_s)
        final_ns = 1 - final_pos - final_neg
        AX[0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (1, -0.6), xycoords='axes fraction')
        
        sem = stats.sem([pos_s, neg_s], axis=1) 

        pt.annotate(AX[idx], 'Pos= %.1f ± %.1f %%' % (100 * final_pos, 100 *sem[0]),
                (1, 0), ha='right', va='top', fontsize=6)
        pt.annotate(AX[idx], 'Neg= %.1f ± %.1f %%' % (100 * final_neg, 100 *sem[1]),
                    (1, -0.2), ha='right', va='top', fontsize=6)
        

    elif type == 'ROI':
        pos_cond_s = np.concatenate(pos_cond_s)
        neg_cond_s = np.concatenate(neg_cond_s)
    
        final_pos = np.mean(pos_cond_s)
        final_neg = np.mean(neg_cond_s)
        final_ns = 1 - final_pos - final_neg
        AX[0].annotate('average over %i ROIs' % np.sum(nROIs),
                               (1, -0.6), xycoords='axes fraction')
        
        pt.annotate(AX[idx], 'Pos= %.1f %%' % (100 * final_pos),
                (1, 0), ha='right', va='top', fontsize=6)
        pt.annotate(AX[idx], 'Neg= %.1f %%' % (100 * final_neg),
                    (1, -0.2), ha='right', va='top', fontsize=6)

    pt.pie(data=[final_pos, final_neg, final_ns],
           ax=AX[idx],
           COLORS=colors)

    AX[idx].set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    
    return 0

def plot_responsiveness2_of_protocol(data_s, AX, idx, p, t_window=1.5, alpha=0.05, type='means', colors = ['red', 'blue', 'grey']):

    #to merge with previous one if possible !!
    pos_s = []
    neg_s = []

    resp_cond_s = []
    pos_cond_s = []
    neg_cond_s = []

    nROIs = []

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
        
        sig_list = []
        val_list = []
        

        print("data nROIS : ", data.nROIs)

        for roi_n in range(data.nROIs):
    
            t0 = max([0, ep.time_duration[0]-t_window])
            stat_test_props = dict(
                interval_pre=[-t_window,0],
                interval_post=[t0, t0+t_window],
                test='ttest',
                sign='both')
            
            roi_summary_data = pre_post_statistics(ep,
                                                   episode_cond = ep.find_episode_cond(),
                                                   response_args = dict(roiIndex=roi_n),
                                                   response_significance_threshold=alpha,
                                                   stat_test_props=stat_test_props,
                                                   repetition_keys=list(ep.varied_parameters.keys()))
            
            sig_list.append(bool(roi_summary_data['significant']))
            val_list.append(roi_summary_data['value'])
        
        nROIs.append(data.nROIs)

        sig_arr = np.array(sig_list)
        val_arr = np.array(val_list)

        #Compute per-ROI positive/negative significance
        resp_cond = sig_arr
        pos_cond = sig_arr & (val_arr > 0)
        neg_cond = sig_arr & (val_arr < 0)

        resp_cond_s.append(resp_cond)
        pos_cond_s.append(pos_cond)
        neg_cond_s.append(neg_cond)

        #Compute per-session proportions
        pos = np.sum(pos_cond) / len(sig_arr)
        neg = np.sum(neg_cond) / len(sig_arr)

        pos_s.append(pos)
        neg_s.append(neg)

    if type== 'means':
        final_pos = np.mean(pos_s)
        final_neg = np.mean(neg_s)
        final_ns = 1 - final_pos - final_neg
        AX.annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (1, -0.6), xycoords='axes fraction')
        
        sem = stats.sem([pos_s, neg_s], axis=1) 

        pt.annotate(AX, 'Pos= %.1f ± %.1f %%' % (100 * final_pos, 100 *sem[0]),
                (1, 0), ha='right', va='top', fontsize=6)
        pt.annotate(AX, 'Neg= %.1f ± %.1f %%' % (100 * final_neg, 100 *sem[1]),
                    (1, -0.2), ha='right', va='top', fontsize=6)
        

    elif type == 'ROI':
        pos_cond_s = np.concatenate(pos_cond_s)
        neg_cond_s = np.concatenate(neg_cond_s)
    
        final_pos = np.mean(pos_cond_s)
        final_neg = np.mean(neg_cond_s)
        final_ns = 1 - final_pos - final_neg
        AX.annotate('average over %i ROIs' % np.sum(nROIs),
                               (1, -0.6), xycoords='axes fraction')
        
        pt.annotate(AX, 'Pos= %.1f %%' % (100 * final_pos),
                (1, 0), ha='right', va='top', fontsize=6)
        pt.annotate(AX, 'Neg= %.1f %%' % (100 * final_neg),
                    (1, -0.2), ha='right', va='top', fontsize=6)
    
    else : 
        print("Give a valid type between 'means' and 'ROI' .")

    print(final_pos, final_neg, final_ns)
    pt.pie(data=[final_pos, final_neg, final_ns],
        ax=AX,
        COLORS=colors)

    AX.set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    
    return 0

def plot_contrast_responsiveness_(keys,
                                  Responsive,
                                  sign='positive',
                                  colors=None,
                                  with_label=True,
                                  fig_args={'right':25}, 
                                  angle = "0.0", 
                                  ylim=[0,100]):

        if colors is None:
            colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

        if type(keys)==str:
                keys, colors = [keys], [colors[0]]

        fig, ax = pt.figure(**fig_args)
        inset = pt.inset(ax, [1.7,0.1,0.5,0.8])

        for i, (key, color) in enumerate(zip(keys, colors)):
                
                x = np.arange(len(np.mean(Responsive[key][angle], axis=0)))+ 0.4*i
                y = np.mean(Responsive[key][angle], axis=0)
                sy = stats.sem(Responsive[key][angle], axis=0)

                print("values to plot : ", y)
                pt.bar(y = y, 
                       sy= sy, 
                       x = x, 
                       width=0.5/len(keys),
                       color=color,
                       ax=ax)
                
                #Gains plot
                contrasts = [0.05,0.18571429,0.32142857,0.45714286,
                             0.59285714, 0.72857143,0.86428571, 1.]
                Gains = []
                for r in Responsive[key][angle]:
                    temp = [r_ / contrast for r_, contrast in zip(r, contrasts)]
                    Gains.append(np.mean(temp))

                pt.violin(Gains, x=i, color=color, ax=inset)
                #pt.bar([np.mean(Gains)], x=[i], color=color, ax=inset, alpha=0.1) #looks confusing to me

                if with_label:
                        annot = i*'\n'+' %.1f$\\pm$%.1f, ' %(\
                               np.mean(Gains), stats.sem(Gains))
                        annot += 'N=%02d %s, ' % (len(Responsive[key][angle]), 'sessions') + key

                pt.annotate(inset, annot, (1., 0.9), va='top', color=color)
                
        
        pt.set_plot(ax, 
                    ylabel='%% responsive \n %s' % sign,
                    xlabel='\ncontrast', 
                    xticks=[0,1,2,3,4,5,6,7], 
                    xticks_labels=[0.05,0.19,0.32,0.46,0.59,0.73,0.86,1.0], 
                    yticks= np.arange(0, ylim[1], 10), 
                    xticks_rotation=90, 
                    ylim=ylim)
        
        
        pt.set_plot(inset, ['left'],
                    title='gain',
                    ylabel='%resp. / contrast')
        
        return fig, ax

def get_vals_resp_vs_param(data_s, p, repetition_keys = ['repeat'], means='ROIs'):
    #old plot_responsiveness2_of_protocol_
    nROIs = 0

    resp_cond_s = []
    pos_cond_s = []
    neg_cond_s = []
    ns_cond_s = []

    for data in data_s:
        ep = EpisodeData(data, 
                         protocol_name=p, 
                         quantities=['dFoF'])
        resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, data.nROIs, alpha=0.05, t_window = 1.5, repetition_keys = repetition_keys)
     
        resp_cond_s.append(resp_cond)
        pos_cond_s.append(pos_cond)
        neg_cond_s.append(neg_cond)
        ns_cond_s.append(~resp_cond)
        
        nROIs += data.nROIs

    if means == 'ROI':
        #average all ROIs 
        resp_cond_s = np.concatenate(resp_cond_s, axis=0) #size (total ROIs x 8 (ex : contrasts))
        pos_cond_s = np.concatenate(pos_cond_s, axis=0)
        neg_cond_s = np.concatenate(neg_cond_s, axis=0)
        ns_cond_s = np.concatenate(ns_cond_s, axis=0)

        final_resp_ = []
        final_pos_ = []
        final_neg_ = []
        final_ns_ = []

        for contrast_i in range(len(resp_cond_s[0])):

            count_resp = np.sum([resp_cond_s[roi_i][contrast_i] for roi_i in range(nROIs)])
            final_resp = (count_resp/nROIs)*100
            final_resp_.append(final_resp)

            count_pos = np.sum([pos_cond_s[roi_i][contrast_i] for roi_i in range(nROIs)])
            final_pos = (count_pos/nROIs)*100
            final_pos_.append(final_pos)

            count_neg = np.sum([neg_cond_s[roi_i][contrast_i] for roi_i in range(nROIs)])
            final_neg = (count_neg/nROIs)*100
            final_neg_.append(final_neg)

            count_ns = np.sum([ns_cond_s[roi_i][contrast_i] for roi_i in range(nROIs)])
            final_ns = (count_ns/nROIs)*100
            final_ns_.append(final_ns)

    elif means == 'session':
        final_resp_sessions = []
        final_pos_sessions = []
        final_neg_sessions = []
        final_ns_sessions = []

        for file_i in range(len(resp_cond_s)):

            nROIs = data_s[file_i].nROIs

            final_resp_ = []
            final_pos_ = []
            final_neg_ = []
            final_ns_ = []

            for contrast_i in range(len(resp_cond_s[file_i][0])):

                count_resp = np.sum([resp_cond_s[file_i][roi_i][contrast_i] for roi_i in range(nROIs)])
                final_resp = (count_resp/nROIs)*100
                final_resp_.append(final_resp)

                count_pos = np.sum([pos_cond_s[file_i][roi_i][contrast_i] for roi_i in range(nROIs)])
                final_pos = (count_pos/nROIs)*100
                final_pos_.append(final_pos)

                count_neg = np.sum([neg_cond_s[file_i][roi_i][contrast_i] for roi_i in range(nROIs)])
                final_neg = (count_neg/nROIs)*100
                final_neg_.append(final_neg)

                count_ns = np.sum([ns_cond_s[file_i][roi_i][contrast_i] for roi_i in range(nROIs)])
                final_ns = (count_ns/nROIs)*100
                final_ns_.append(final_ns)
            
            final_resp_sessions.append(final_resp_)
            final_pos_sessions.append(final_pos_)
            final_neg_sessions.append(final_neg_)
            final_ns_sessions.append(final_ns_)
        
        final_pos_ = np.mean(final_pos_sessions, axis=0)
        final_neg_ = np.mean(final_neg_sessions, axis=0)
        final_ns_  = np.mean(final_ns_sessions, axis=0)

    return final_pos_, final_neg_, final_ns_

def plot_resp_vs_param(data_s, p, AX, test = "angle", repetition_keys = ['repeat'], means = 'session'):

    pos, neg, ns = get_vals_resp_vs_param(data_s, p, repetition_keys = repetition_keys, means = means)

    ep = EpisodeData(data_s[0], protocol_name=p, quantities=['dFoF'])
    x = ep.varied_parameters[test]

    for x, vals, color in zip([x,x,x],[pos, neg, ns],["red", "blue", "grey"]):
        pt.plot(x, vals, ax=AX, color = color)
       
    pt.set_plot(ax = AX, 
                ylabel='Responsiveness (%)', 
                yticks=np.arange(0, 105, 10),
                xticks = x,
                xlabel=test, 
                xticks_labels=[f"{xi:.2f}" for xi in x],
                xticks_rotation=90,
                fontsize = 15,
                ylim=[0,100])
    
    return 0
