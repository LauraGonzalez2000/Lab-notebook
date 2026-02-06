# %% [markdown]
# # Responsiveness

#%%
#why are Data created??
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

sys.path += ['..']
from General_summary.Generate_PDF_summary_opti import plot_responsiveness2_per_protocol, plot_responsiveness_per_protocol

#%%
############################################################################
######################## FUNCTIONS #########################################
############################################################################ 

def study_responsiveness(SESSIONS, index, protocol):

    filename = SESSIONS['files'][index]
    data = Data(filename, verbose=False)
    data.build_dFoF()
   
    protocol = protocol
    stat_test_props = dict(interval_pre=[-1.,0],                                   
                        interval_post=[1.,2.],                                   
                        test='ttest')

    ep = EpisodeData(data,
                    protocol_name=protocol,
                    quantities=['dFoF'])

    print("dFoF shape : ", ep.dFoF.shape)
    print("varied parameters : ", ep.varied_parameters)

    values = []
    significance = []
    colors = []

    for roi_n in range(data.nROIs):
        ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)

        
        roi_summary_data = pre_post_statistics(ep,
                                           episode_cond = ep.find_episode_cond(),
                                           response_args = dict(roiIndex=roi_n),
                                           response_significance_threshold=0.05,
                                           stat_test_props=stat_test_props,
                                           repetition_keys=['repeat', 'angle', 'contrast'])
        
        if roi_summary_data['significant']: 
            if roi_summary_data['value'] < 0: color = 'red'
            else: color = 'green'
            colors.append(color)
        else: 
            if roi_summary_data['value'] < 0: color = 'pink'
            else: color = 'lime'
            colors.append(color)

        values.append(roi_summary_data['value'].flatten())
        significance.append(roi_summary_data['significant'].flatten())

    fig, AX = plt.subplots(1, 1, figsize=(1, 1))
    x= np.arange(0,len(values),1)
    y = [float(value) for value in values]
    AX.bar(x, y, color=colors)
    AX.set_xlabel('ROI #')
    AX.set_ylabel('Responsiveness')
    AX.set_title(f'Session #{index}')
    print(significance)
    true_indexes = [i for i, val in enumerate(significance) if val]
    false_indexes = [i for i, val in enumerate(significance) if not val]
    print(true_indexes)
    print(f'{len(true_indexes)} significant ROI out of {len(significance)} ROIs')
    return 0

def study_responsiveness_all(SESSIONS, protocol):

    Colors = []
    Values = []
    Significance = []

    for rec in range(len(SESSIONS['files'])):
        filename = SESSIONS['files'][rec]
        data = Data(filename, verbose=False)
        data.build_dFoF()

        protocol = "static-patch"
        stat_test_props = dict(interval_pre=[-1.,0],                                   
                            interval_post=[1.,2.],                                   
                            test='ttest')
        
        ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)
        
        values = []
        significance = []
        colors = []

        for roi_n in range(data.nROIs):
            
            #summary_data = ep.compute_summary_data(stat_test_props,
            #                                    exclude_keys=['repeat', 'angle', 'contrast'],
            #                                    response_significance_threshold=0.05,
            #                                    response_args=dict(roiIndex=roi_n))
            
            roi_summary_data = pre_post_statistics(ep,
                                                   episode_cond = ep.find_episode_cond(),
                                                   response_args = dict(roiIndex=roi_n),
                                                   response_significance_threshold=0.05,
                                                   stat_test_props=stat_test_props,
                                                   repetition_keys=['repeat', 'angle', 'contrast'])
            
            if roi_summary_data['significant']: 
                if roi_summary_data['value'] < 0: color = 'red'
                else: color = 'green'
                colors.append(color)
            else: 
                if roi_summary_data['value'] < 0: color = 'pink'
                else: color = 'lime'
                colors.append(color)

            values.append(roi_summary_data['value'].flatten())
            significance.append(roi_summary_data['significant'].flatten())
        
        Colors.append(colors)
        Values.append(values)
        Significance.append(significance)

    
    fig, AX = plt.subplots(5, 5, figsize=(9, 9))

    i,j = 0,0

    for rec in range(len(SESSIONS['files'])):

        x= np.arange(0,len(Values[rec]),1)
        y = [float(value) for value in Values[rec]]    

        AX[i][j].bar(x, y, color=Colors[rec])
        AX[i][j].set_xlabel('ROI #')
        AX[i][j].set_ylabel('Responsiveness')
        AX[i][j].set_title(f'Session #{rec}')

        # ---- PIE CHART ----
        # Count responsive cells
        n_total = len(Significance[rec])
        true_indexes = [i for i, val in enumerate(Significance[rec]) if val]
        false_indexes = [i for i, val in enumerate(Significance[rec]) if not val]
        
        excit_indexes = sum(1 for v in true_indexes if v > 0)
        inhib_indexes = sum(1 for v in true_indexes if v < 0)
        n_nonresponsive = len(false_indexes)
        n_total = len(Significance[rec])
        
        
        pie_counts = [excit_indexes, inhib_indexes, n_nonresponsive]
        pie_colors = ['g', 'r', 'gray']
        
        # Add as inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(AX[i][j], width="30%", height="30%", loc='upper right')
        ax_inset.pie(pie_counts, colors=pie_colors)#, autopct='%1.0f%%')
        #ax_inset.set_title('Responsive cells', fontsize=8)

        #update position next graph
        if j<4:
            j+=1
        else: 
            i+=1
            j=0

    AX = AX.flatten()
    for idx in range(len(SESSIONS['files']), len(AX)):
        fig.delaxes(AX[idx])
    fig.tight_layout()
    return 0

def plot_responsiveness_per_protocol(ep, nROIs, AX, idx, p, Resp_ROI_dict):
    
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(nROIs):

        t0 = max([0, ep.time_duration[0]-1.5])

        stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

        #roi_summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
        #                                           #exclude_keys=['repeat'],
        #                                           exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
        #                                           response_significance_threshold=0.05,
        #                                           response_args=dict(roiIndex=roi_n))
        roi_summary_data = pre_post_statistics(ep,
                                           episode_cond = ep.find_episode_cond(),
                                           response_args = dict(roiIndex=roi_n),
                                           response_significance_threshold=0.05,
                                           stat_test_props=stat_test_props,
                                           repetition_keys=list(ep.varied_parameters.keys()))
        
        session_summary['significant'].append(bool(roi_summary_data['significant']))
        session_summary['value'].append(roi_summary_data['value'])
        print("roi :", roi_n, ", resp : ",  bool(roi_summary_data['significant']), ", value : ",roi_summary_data['value'], "\n")
        if bool(roi_summary_data['significant'])==False:
            category = 'NS'
        else: 
            if roi_summary_data['value']>0:
                category = "Positive"
            else: 
                category = "Negative"

        Resp_ROI_dict[f"ROI_{roi_n}"][p] = category

    resp_cond = np.array(session_summary['significant'])                     
    pos_cond = resp_cond & ([session_summary['value'][i]>0 for i in range(len(session_summary['value']))])
    neg_cond = resp_cond & ([session_summary['value'][i]<0 for i in range(len(session_summary['value']))])

    print(f'Protocol {p} : {sum(resp_cond)} significant ROI ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) out of {len(session_summary['significant'])} ROIs')

    pos_frac = np.sum(pos_cond)/nROIs
    neg_frac = np.sum(neg_cond)/nROIs
    ns_frac = 1-pos_frac-neg_frac

    colors = ['green', 'red', 'grey']

    pt.pie(data=[pos_frac, neg_frac, ns_frac], ax = AX[idx], COLORS = colors)#, pie_labels = ['%.1f%%' % (100*pos),'%.1f%%' % (100*neg), '%.1f%%' % (100*ns)] )
    
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    pt.annotate(AX[idx], '+ resp=%.1f%% ' % (100*pos_frac), (1, 0), ha='right', va='top')
    pt.annotate(AX[idx], '- resp=%.1f%%' % (100*neg_frac), (1, -0.2), ha='right', va='top')
    pt.annotate(AX[0], f'{nROIs} ROIs', (1, -0.4), ha='right', va='top')
    return 0

def plot_responsiveness2_per_protocol(data_s, AX, idx,  p, type='means', behavior_split=False):
    pos_s = []
    neg_s = []

    resp_cond_s = []
    pos_cond_s = []
    neg_cond_s = []

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
        
        sig_list = []
        val_list = []
        nROIs = []

        for roi_n in range(data.nROIs):

            t0 = max([0, ep.time_duration[0]-1.5])
            stat_test_props = dict(
                interval_pre=[-1.5,0],
                interval_post=[t0, t0+1.5],
                test='ttest',
                sign='both')
            
            roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=0.05,
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

    print(final_pos, final_neg, final_ns)
    pt.pie(data=[final_pos, final_neg, final_ns],
        ax=AX[idx],
        COLORS=['green', 'red', 'grey'])

    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    
    return 0

#%% Load data
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Vision-survey', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

data_s = []
for idx, filename in enumerate(SESSIONS['files']):

    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)

#%%
################ PIE CHARTS RESPONSIVENESS PER PROTOCOL ###################################################
#protocols = ["static-patch", "drifting-gratings", "Natural-Images-4-repeats"]
#protocols = ['static-patch', 
#             'drifting-gratings', 
#             'looming-stim',
#             'Natural-Images-4-repeats', 
#             'moving-dots']#, 
#             'random-dots']
#protocols = ['static-patch', 'looming-stim']

protocols = ['static-patch', 
             'drifting-grating', #my data!!! "s" for Yann's data
             'looming-stim',
             'Natural-Images-4-repeats', 
             'moving-dots']


fig, AX  = pt.figure(axes = (len(protocols),1))

for idx, p in enumerate(protocols):
    plot_responsiveness2_per_protocol(data_s, AX, idx, p, type='ROI', behavior_split=False)

#%%
############################################################################################################
############################## PLOT RESPONSIVENESS FOR GIVEN PROTOCOL ######################################
##################################### PER SESSION and PER ROI ##############################################
############################################################################################################

#%%
########################################### STATIC PATCH     ###############################################
protocol="static-patch"
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)

#%%
########################################### DRIFTING GRATING ###############################################
#%% DATA ALL
protocol="drifting-grating"
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%% YANN'S DATA
protocol="drifting-gratings"
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)

#%%
########################################### NATURAL IMAGES ###############################################
protocol='Natural-Images-4-repeats'
#%% DATA ALL
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%% YANN'S DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)


############################################################################################################
############################################ Plot per protocol #############################################
############################################################################################################
#%%  ERROR TO FIX
'''
def responsiveness_sessions_vs_protocols(SESSIONS, protocols):
    Values_m = []
    X = []

    for p, protocol in enumerate(protocols):
        
        values_per_file = []
        significance_per_file = []
        
        for f, filename in enumerate(SESSIONS['files']):
            data = Data(filename, verbose=False)
            data.build_dFoF()
            stat_test_props = dict(interval_pre=[-1.,0],                                   
                                interval_post=[1.,2.],                                   
                                test='ttest')
            ep = EpisodeData(data,
                            protocol_name=protocol,
                            quantities=['dFoF'], 
                            verbose=False)
        
            #summary_data = ep.compute_summary_data(stat_test_props,
            #                                    exclude_keys=['repeat', 'angle', 'contrast'],
            #                                    response_significance_threshold=0.05,
            #                                    response_args={})
            summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = {},
                                               response_significance_threshold=0.05,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=['repeat', 'angle', 'contrast'])

            values_per_file.append(summary_data['value'].flatten())
            significance_per_file.append(summary_data['significant'].flatten())
        
        X.append(np.arange(0,len(SESSIONS['files']),1))
        Values_m.append(values_per_file)

    # ## plot
   
    i,j = 0,0

    fig, AX = plt.subplots(2, 4, figsize=(10, 5))

    for p, protocol in enumerate(protocols):
        try: 
            AX[i][j].bar(X[p], np.array(Values_m[p]).flatten())
        except Exception as e:
            AX[i][j].bar(X[p], np.array(np.mean(Values_m[p], axis=1)).flatten())
        AX[i][j].axhline(0.0, c='black', linewidth = 0.5)
        AX[i][j].set_xlabel('File #')
        AX[i][j].set_title(f'{protocol}')
        AX[i][j].set_xticks(X[p])

        if j==0:
            AX[i][j].set_ylabel('responsiveness')

        if protocol!='Natural-Images-4-repeats':
            AX[i][j].set_title(f'{protocol}')
        else: 
            AX[i][j].set_title('Natural-Images')

        #update position next graph
        if j<3:
            j+=1
        else: 
            i+=1
            j=0

    AX = AX.flatten()
    n_files = len(protocols)
    print(n_files)
    for idx in range(n_files, len(AX)):
        fig.delaxes(AX[idx])
    fig.tight_layout()
    return 0
# MY DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
protocols = ['static-patch', 'drifting-grating' ,'looming-stim',
             'Natural-Images-4-repeats','moving-dots',
             'drifting-surround','quick-spatial-mapping']
responsiveness_sessions_vs_protocols(SESSIONS=SESSIONS, protocols=protocols)
#######################################################################################################################
# YANN's DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
protocols = ['static-patch', 'drifting-gratings', 'looming-stim',
              'Natural-Images-4-repeats', 'moving-dots', 
              'random-dots']
responsiveness_sessions_vs_protocols(SESSIONS=SESSIONS, protocols=protocols)
########################################################################################################################
'''

