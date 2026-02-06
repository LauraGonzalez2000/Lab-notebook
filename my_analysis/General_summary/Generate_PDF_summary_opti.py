# %% [markdown]
# # Generate PNG summary

#%%
import os, sys
import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
from physion.utils import plot_tools as pt
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.raw import plot as plot_raw

from scipy import stats
base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'NWBs')
import random
from PDF_layout import PDF, PDF2, PDF3, PDF3_
from matplotlib.backends.backend_pdf import PdfPages

from utils_.General_overview_episodes import plot_dFoF_per_protocol, plot_dFoF_per_protocol2
from Visual_Properties_analysis.Responsiveness import plot_responsiveness_per_protocol, plot_responsiveness2_per_protocol

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import time


#%%
#######################################################################################################
##################################### FUNCTIONS #######################################################
####################################################################################################### 
def find_available_settings(data, debug=False):

    settings = {'Locomotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': '#1f77b4'},
                        'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                        'Pupil': {'fig_fraction': 2,
                                                  'subsampling': 1,
                                                  'color': '#d62728'},
                        'CaImaging': {'fig_fraction': 10,
                                                   'subsampling': 1,
                                                   'subquantity': 'dF/F',
                                                   'color': '#2ca02c'}}

    

    attributes = ['facemotion', 'pupil_diameter', 'dFoF']
    
    missing = [attr for attr in attributes if not hasattr(data, attr)]  # for objects

    if debug:
        if missing:
            print(f"Missing attributes: {missing}")
        else:
            print("All attributes exist")
    
    if missing==['pupil_diameter']:
        if debug:
            print("only pupil diameter missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': '#1f77b4'},
                        'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                        'CaImaging': {'fig_fraction': 10,
                                                   'subsampling': 1,
                                                   'subquantity': 'dF/F',
                                                   'color': '#2ca02c'}}
    
    if missing==['facemotion']:
        if debug:
            print("only pupil diameter missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                        'Pupil': {'fig_fraction': 2,
                                          'subsampling': 1,
                                          'color': '#d62728'},
                        'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'color': '#2ca02c'}}
        
    if missing==['dFoF']:
        if debug:
            print("only Ca imaging missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                    'Pupil': {'fig_fraction': 2,
                                          'subsampling': 1,
                                          'color': '#d62728'}}
                        
    
    if missing==['facemotion', 'pupil_diameter']:
        if debug:
            print('facemotion and pupil diameter missing')
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'color': '#2ca02c'}}
        
    return settings

def figure_to_array(fig):
            """Convert a matplotlib Figure to a NumPy array"""
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            fig_arr = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            return fig_arr

def plot_barplot_per_protocol(ep, AX, idx, p, subplots_n):
        
    t0 = max([0, ep.time_duration[0]-1.5])
    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                            interval_post=[t0, t0+1.5],                                   
                            test='ttest', 
                            sign='both')
    
    #summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
    #                                        exclude_keys=['repeat'],
    #                                        response_significance_threshold=0.05,
    #                                        response_args={})
    summary_data = pre_post_statistics(ep,
                                       episode_cond = ep.find_episode_cond(),
                                       response_args = {},
                                       response_significance_threshold=0.05,
                                       stat_test_props=stat_test_props,
                                       repetition_keys=['repeat'])
    
    mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

    # Pad or truncate to 9 elements
    target_len = subplots_n #generalize
    if len(mean_vals) < target_len:
        mean_vals.extend([np.nan] * (target_len - len(mean_vals)))
    else:
        mean_vals = mean_vals[:target_len]


    x = np.arange(target_len) #generalize
    AX[idx].bar(x, mean_vals, alpha=0.8, capsize=4)
    AX[idx].set_xticks(x)
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    AX[idx].axhline(0, color='black', linewidth=0.8)

    if idx==0:
        AX[0].set_ylabel('variation dFoF')

    return 0

def get_roiIndex(data, type='pos'):
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(data.nROIs):
        ep = EpisodeData(data,
                        protocol_name='Natural-Images-4-repeats',
                        quantities=['dFoF'], 
                        verbose=False)

        t0 = max([0, ep.time_duration[0]-1.5])
        stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

        #roi_summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
        #                                            #exclude_keys=['repeat'],
        #                                            exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
        #                                            response_significance_threshold=0.05,
        #                                            response_args=dict(roiIndex=roi_n))
        
        roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=0.05,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=list(ep.varied_parameters.keys()))
        
        session_summary['significant'].append(bool(roi_summary_data['significant']))
        session_summary['value'].append(roi_summary_data['value'])

    resp_cond = session_summary['significant']
    pos_cond = resp_cond.copy()
    neg_cond = resp_cond.copy()
 
    for i in range(len(resp_cond)):
        if resp_cond[i]==False:
            pos_cond[i]=False
            neg_cond[i]=False
        else: 
            if session_summary['value'][i]>0:
                pos_cond[i]=True
                neg_cond[i]=False
            else: 
                pos_cond[i]=False
                neg_cond[i]=True

    pos_roi = np.where(pos_cond)[0]
    neg_roi = np.where(neg_cond)[0]
    ns_roi = np.where(~np.array(resp_cond))[0]

    found = True

    if type=='pos':
        if len(pos_roi) > 0:
            roiIndex = random.choice(pos_roi)
        else:
            print("No positive ROIs found — choosing from non-significant set instead.")
            found = False
            roiIndex = random.choice(ns_roi)

    elif type=='neg':
        if len(neg_roi) > 0:
            roiIndex = random.choice(neg_roi)
        else:
            print("No negative ROIs found — choosing from non-significant set instead.")
            roiIndex = random.choice(ns_roi)
            found = False

    elif type=='ns':
        roiIndex = random.choice(ns_roi) 

    return roiIndex, found

def generate_figures(data_s, cell_type='nan', subplots_n=9, data_type = 'Sofia'):
    start_time = time.time()
    # Generate figures per session 

    for idx, data in enumerate(data_s):

        dict_annotation = {
            'name': data.filename,
            'Subject_ID': data.metadata['subject_ID'],
            'protocol': data.metadata['protocol']
        }

        settings = find_available_settings(data)
        protocols = [p for p in data.protocols if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

        ################################# FIGURES ###############################################################
        
        fig1, AX1 = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[0])
        show_CaImaging_FOV(data, key='max_proj',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[1])
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2,  ax=AX1[2])
        
        if hasattr(data, "dFoF") and data.dFoF is not None and len(data.dFoF) > 0:
            '''
            data_cropped = data
            
            max_n = 10
            n_rois_total = len(data_cropped.dFoF)

            if n_rois_total > max_n:
                idx = np.random.choice(n_rois_total, max_n, replace=False)
                data_cropped.dFoF = data_cropped.dFoF[idx, :]
            '''
            if data_type=='Sofia':
                fig2, _ = plot_raw(data, 
                                    tlim=[0, data.t_dFoF[-1]], 
                                    settings=settings, 
                                    figsize=(9,3),
                                    zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                                ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])],
                                    grey=True, 
                                    black=True,
                                    grey_co=[638, 1238],
                                    black_co=[466, 586])
            else:
                fig2, _ = plot_raw(data, 
                                    tlim=[0, data.t_dFoF[-1]], 
                                    settings=settings, 
                                    figsize=(9,3),
                                    zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                                ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
            
            fig3, _ = plot_raw(data, tlim=[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]],
                            settings=settings, figsize=(9,3))
            
            fig4, _ = plot_raw(data, tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]],
                            settings=settings, figsize=(9,3))
            
              
        fig5, _ = plot_dFoF_per_protocol(data_s=[data], protocols=protocols, subplots_n=subplots_n)
        
        roiIndex, found = get_roiIndex(data, type='pos')
        fig6, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        roiIndex, found = get_roiIndex(data, type='neg')
        fig7, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        roiIndex, found = get_roiIndex(data, type='ns')
        fig8, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        fig9, AX9 = pt.figure(axes = (len(protocols),1))
        fig10, AX10 = pt.figure(axes = (len(protocols),1))
  
        nROIs = data.nROIs

        for idx, p in enumerate(protocols):
            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
            plot_responsiveness_per_protocol(ep, nROIs, AX9, idx, p)
            plot_barplot_per_protocol(ep, AX10, idx, p, subplots_n)
        
        fig1 = figure_to_array(fig1)
        fig2 = figure_to_array(fig2)
        fig3 = figure_to_array(fig3)
        fig4 = figure_to_array(fig4)
        fig5 = figure_to_array(fig5)
        fig6 = figure_to_array(fig6)
        fig7 = figure_to_array(fig7)
        fig8 = figure_to_array(fig8)
        fig9 = figure_to_array(fig9)
        fig10 = figure_to_array(fig10)

        create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type)

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
        
    return 0    

def create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type):
    try: 
        pdf1 = PDF()
        pdf1.fill_PDF(dict_annotation, fig1, fig2, fig3, fig4)
        fig_p1 = pdf1.fig

        pdf2 = PDF2()
        pdf2.fill_PDF2(fig5, fig10)
        fig_p2 = pdf2.fig

        pdf3 = PDF3()
        pdf3.fill_PDF3(fig6, fig7, fig8, fig9)
        fig_p3 = pdf3.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/{os.path.splitext(dict_annotation['name'])[0]}_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2
                pdf.savefig(fig_p3, dpi=300, bbox_inches="tight")  # Page 3

        print("Individual PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual PDF file : {e}")

    return 0

def plot_barplot2_per_protocol(data_s, AX,idx,  p, subplots_n):
    mean_vals_s = []  # store per-session mean responses

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])

        t0 = max([0, ep.time_duration[0] - 1.5])
        stat_test_props = dict(
            interval_pre=[-1.5, 0],
            interval_post=[t0, t0 + 1.5],
            test='ttest',
            sign='both')

        #summary_data = ep.compute_summary_data(
        #    stat_test_props=stat_test_props,
        #    exclude_keys=['repeat'],
        #    response_significance_threshold=0.05,
        #    response_args={})
        
        summary_data = pre_post_statistics(ep,
                                           episode_cond = ep.find_episode_cond(),
                                           response_args = {},
                                           response_significance_threshold=0.05,
                                           stat_test_props=stat_test_props,
                                           repetition_keys=['repeat'])

        # Extract ROI mean values
        mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

        # Pad/truncate to 5 elements
        target_len = subplots_n
        mean_vals = (mean_vals + [np.nan] * target_len)[:target_len]

        mean_vals_s.append(mean_vals)

    # Compute session-aggregated mean and SEM
    values = np.nanmean(mean_vals_s, axis=0)
    yerr = stats.sem(mean_vals_s, axis=0, nan_policy='omit')

    # Plot directly
    x = np.arange(len(values))
    AX[idx].bar(
        x, values, yerr=yerr,
        alpha=0.8, capsize=0,
        error_kw=dict(linewidth=0.6)
    )
    AX[idx].set_xticks(x)
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    AX[idx].axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX[0].set_ylabel('variation dFoF')
    
    return 0

def generate_figures_GROUP(data_s, subplots_n):
    start_time = time.time()  

    protocols = [p for p in data_s[0].protocols 
                        if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

    fig1, _     = plot_dFoF_per_protocol(data_s=data_s, protocols=protocols)
    elapsed = time.time() - start_time
    print(f"Fig 1 ok: {elapsed:.2f} seconds")

    fig2, AX2  = pt.figure(axes = (len(protocols),1))
    fig3, AX3  = pt.figure(axes = (len(protocols),1))
    fig4, AX4 = pt.figure(axes = (len(protocols),1))
   
    for idx, p in enumerate(protocols):
        plot_responsiveness2_per_protocol(data_s, AX2, idx, p, type='ROI')
        plot_responsiveness2_per_protocol(data_s, AX3, idx, p, type='means')
        plot_barplot2_per_protocol(data_s, AX4, idx, p, subplots_n)

    elapsed = time.time() - start_time
    print(f"Fig 2-3-4 ok: {elapsed:.2f} seconds")

    fig1 = figure_to_array(fig1)
    fig2 = figure_to_array(fig2)
    fig3 = figure_to_array(fig3)
    fig4 = figure_to_array(fig4)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")
    return fig1, fig2, fig3, fig4

def create_group_PDF(fig1, fig2, fig3, fig4, cell_type):
    try: 
        
        pdf1 = PDF2()
        pdf1.fill_PDF2(fig1, fig4)
        fig_p1 = pdf1.fig
        
        pdf2 = PDF3_()
        pdf2.fill_PDF3(fig2, fig3)
        fig_p2 = pdf2.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/GROUP_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2

        print("GROUP PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating GROUP PDF file : {e}")

    return 0

##################################################################################################################
##################################################################################################################
##################################################################################################################
# %% [markdown]
# # Generate final figures
#%% [markdown]
# ## YANN DATASET

#%%
'''
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
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
'''
#%% [markdown]
# ## All individual files
#%%
#generate_figures(data_s, cell_type='NDNF_YANN', subplots_n=5, data_type = 'Yann')
#%% [mardown]
# ## GROUPED ANALYSIS
#%%
'''
protocols = [p for p in data_s[0].protocols 
                        if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]


#fig2, AX2  = pt.figure(axes = (len(protocols),1))
fig1, _     = plot_dFoF_per_protocol(data_s=data_s, protocols=protocols)

#for idx, p in enumerate(protocols):
#        plot_responsiveness2_per_protocol(data_s, AX2, idx, p, type='ROI')

'''
#%%
#fig1, fig2, fig3, fig4 = generate_figures_GROUP(data_s, subplots_n=5)
#create_group_PDF(fig1, fig2, fig3, fig4, 'NDNF_YANN')

##################################################################################################################################
##################################################################################################################################

#%% [markdown]
# ## NDNF CRE BATCH 1



#%%
'''
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_run')
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
'''
#%% [markdown]
# ## All individual files
#%%

#generate_figures(data_s, cell_type='NDNF', subplots_n=5, data_type = 'Sofia')

#%% [mardown]
# ## GROUPED ANALYSIS
#%%

#fig1, fig2, fig3, fig4 = generate_figures_GROUP(data_s, subplots_n=5)
#create_group_PDF(fig1, fig2, fig3, fig4, 'NDNF')
