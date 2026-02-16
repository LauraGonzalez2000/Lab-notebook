# %% [markdown]
# # Generate PNG summary

#%%
import os, sys, tempfile
import numpy as np


sys.path += ['../../physion/src'] # add src code directory for physion
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.raw import plot as plot_raw
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

#orientation tunning
from physion.analysis.protocols.orientation_tuning import plot_orientation_tuning_curve, \
                                                          plot_selectivity, \
                                                          plot_responsiveness

from scipy import stats
import random
sys.path += ['..']
from Orientation_Tuning import compute_tunings2

from PDF_layout import PDF, PDF2, PDF3, PDF_angle_contrast, PDF_ori_tuning
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import time
from utils_.General_overview_episodes import plot_dFoF_of_protocol, plot_dFoF_of_protocol2

#%% 
### UTILS FUNCTIONS ###
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
    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    # drop alpha channel → RGB
    fig_arr = buf[:, :, :3].copy()

    return fig_arr

def calc_responsiveness(ep, nROIs):
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(nROIs):

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
                                               repetition_keys=['repeat'])

        session_summary['significant'].append(bool(roi_summary_data['significant'][0]))
        session_summary['value'].append(roi_summary_data['value'][0])

    resp_cond = np.array(session_summary['significant'])                  
    pos_cond = resp_cond & ([session_summary['value'][i]>0 for i in range(len(session_summary['value']))])
    neg_cond = resp_cond & ([session_summary['value'][i]<0 for i in range(len(session_summary['value']))])

    print(f"{sum(resp_cond)} significant ROI \n ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) \n out of {len(session_summary['significant'])} ROIs")

    return resp_cond, pos_cond, neg_cond

def compute_responsiveness(ep, nROIs, alpha=0.05, window=1.5):
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

    return dict(
        pos_frac=np.sum(pos) / nROIs,
        neg_frac=np.sum(neg) / nROIs,
        ns_frac=np.sum(ns) / nROIs,
        n_pos=np.sum(pos),
        n_neg=np.sum(neg),
        n_sig=np.sum(significant),
        nROIs=nROIs
    )

def get_roiIndex(data, type='pos', protocol="Natural-Images-4-repeats"):
    
    session_summary = {'significant':[], 'value':[]}

    ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)
    t0 = max([0, ep.time_duration[0]-1.5])
    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

    for roi_n in range(data.nROIs):
        
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
        if  resp_cond[i]==True:
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
        if len(ns_roi) >0: 
            roiIndex = random.choice(ns_roi) 
        else : 
            print("No ns ROIs found — choosing from significant set instead.")
            try : 
                roiIndex = random.choice(pos_roi)
                found = False
            except : 
                roiIndex = random.choice(neg_roi)
                found = False


    return roiIndex, found

### PLOT functions ###
def plot_protocol_responsiveness(ep, nROIs, AX, protocol="", idx=None, colors=['green', 'red', 'grey']):
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
    ax.set_title(protocol)
    pt.annotate(ax, f"+ resp={100*resp['pos_frac']:.1f}%", (1, 0), ha='right', va='top')
    pt.annotate(ax, f"- resp={100*resp['neg_frac']:.1f}%", (1, -0.2), ha='right', va='top')
    pt.annotate(ax, f"{resp['nROIs']} ROIs", (1, -0.4), ha='right', va='top')

def plot_protocol_ddFoF(ep, AX, idx, protocol="", subplots_n=16):

    ax = AX[idx] if idx is not None else AX

    t0 = max([0, ep.time_duration[0]-1.5])
    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                            interval_post=[t0, t0+1.5],                                   
                            test='ttest', 
                            sign='both')
    
    summary_data = pre_post_statistics(ep,
                                       episode_cond = ep.find_episode_cond(),
                                       response_args = {},
                                       response_significance_threshold=0.05,
                                       stat_test_props=stat_test_props,
                                       repetition_keys=['repeat'],
                                       nMin_episodes=5,
                                       multiple_comparison_correction=True,
                                       loop_over_cells=False,
                                       verbose=True)
    
    mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

    target_len = subplots_n #generalize
    if len(mean_vals) < target_len:
        mean_vals.extend([np.nan] * (target_len - len(mean_vals)))
    else:
        mean_vals = mean_vals[:target_len]


    x = np.arange(target_len) #generalize
    ax.bar(x, mean_vals, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_title(f"{protocol.replace('Natural-Images-4-repeats','natural-images')}")
    ax.axhline(0, color='black', linewidth=0.8)

    if idx==0:
        AX[0].set_ylabel('variation dFoF')

### GENERATE FINAL FIGURES ###
def generate_figures(data_s, subplots_n=16):
    start_time = time.time()

    for data in data_s:
        dict_annotation = {'name': data.filename,
                           'Subject_ID': data.metadata['subject_ID'],
                           'protocol': data.metadata['protocol']}

        settings = find_available_settings(data)
        protocols = [p for p in data.protocols if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

        ################################# FIGURES ###############################################################
        
        fig1, AX1 = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[0])
        show_CaImaging_FOV(data, key='max_proj',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[1])
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2,  ax=AX1[2])
        

        if hasattr(data, "dFoF") and data.dFoF is not None and len(data.dFoF) > 0:
            zoom_area = [((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                         ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])]
            print(zoom_area)
            fig2, _ = plot_raw(data, 
                                tlim=[0, data.t_dFoF[-1]], 
                                settings=settings, 
                                figsize=(9,3),
                                zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                                ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
        
            fig3, _ = plot_raw(data, 
                               tlim = [(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]],
                               settings=settings, 
                               figsize=(9,3))
            
            fig4, _ = plot_raw(data, 
                               tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]],
                               settings=settings, 
                               figsize=(9,3))
            
          
        fig5, _ = plot_dFoF_of_protocol(data_s=[data], protocol=protocols[0], subplots_n=subplots_n)
        
        fig10, AX10 = pt.figure(axes = (len(protocols),1),figsize=(1.4,8) )

        ep = EpisodeData(data, protocol_name=protocols[0], quantities=['dFoF'])
        plot_protocol_ddFoF(ep=ep, AX=AX10, idx=None, protocol=protocols[0], subplots_n=subplots_n)
        
        roiIndex, found = get_roiIndex(data, type='pos', protocol=data.protocols[0])
        fig6, _ = plot_dFoF_of_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        
        roiIndex, found = get_roiIndex(data, type='neg', protocol=data.protocols[0])
        fig7, _ = plot_dFoF_of_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        
        roiIndex, found = get_roiIndex(data, type='ns', protocol=data.protocols[0])
        fig8, _ = plot_dFoF_of_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        
        fig9, AX9 = pt.figure(axes = (len(protocols),1))
        nROIs = data.nROIs
        plot_protocol_responsiveness(ep, nROIs, AX=AX9, protocol=protocols[0])
        
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

        #create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type)

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
        
    return dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10    

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

        output_path = f"C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/{os.path.splitext(dict_annotation['name'])[0]}_summary.pdf"

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2
                pdf.savefig(fig_p3, dpi=300, bbox_inches="tight")  # Page 3

        print("Individual PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual PDF file : {e}")

    return 0

##################################################################################################################
##################################################################################################################

def plot_responsiveness2_per_protocol(data_s, AX, idx, p, type='means'):
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

            #ep.compute_summary_data(stat_test_props=stat_test_props,
            #                                           exclude_keys=list(ep.varied_parameters.keys()),
            #                                           response_significance_threshold=0.05,
            #                                           response_args=dict(roiIndex=roi_n))

            sig_list.append(bool(roi_summary_data['significant'][0]))
            val_list.append(roi_summary_data['value'][0])
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

    AX[idx].set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    
    return 0

def plot_responsiveness2_of_protocol(data_s, AX, idx, p, type='means'):
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
        COLORS=['green', 'red', 'grey'])

    AX.set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    
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
    AX[idx].set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    AX[idx].axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX[0].set_ylabel('variation dFoF')
    
    return 0

def plot_barplot2_of_protocol(data_s, AX, idx,  p, subplots_n):
    mean_vals_s = []  # store per-session mean responses

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])

        t0 = max([0, ep.time_duration[0] - 1.5])
        stat_test_props = dict(
            interval_pre=[-1.5, 0],
            interval_post=[t0, t0 + 1.5],
            test='ttest',
            sign='both')

        summary_data = pre_post_statistics(ep,
                                           episode_cond = ep.find_episode_cond(),
                                           response_args = {},
                                           response_significance_threshold=0.05,
                                           stat_test_props=stat_test_props,
                                           repetition_keys=['repeat'])
        
        # Extract ROI mean values
        print("summary data value \n", summary_data['value'])
        mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]
     
        # Pad/truncate to 5 elements
        target_len = subplots_n
        mean_vals = (mean_vals + [np.nan] * target_len)[:target_len]

        if p == 'ff-gratings-8orientation-2contrasts-15repeats':
            #reorder mean_vals
            mean_vals_ = np.zeros(16)
            mean_vals_[0] = mean_vals[0]
            mean_vals_[1] = mean_vals[2]
            mean_vals_[2] = mean_vals[4]
            mean_vals_[3] = mean_vals[6]
            mean_vals_[4] = mean_vals[8]
            mean_vals_[5] = mean_vals[10]
            mean_vals_[6] = mean_vals[12]
            mean_vals_[7] = mean_vals[14]

            mean_vals_[8] = mean_vals[1]
            mean_vals_[9] = mean_vals[3]
            mean_vals_[10] = mean_vals[5]
            mean_vals_[11] = mean_vals[7]
            mean_vals_[12] = mean_vals[9]
            mean_vals_[13] = mean_vals[11]
            mean_vals_[14] = mean_vals[13]
            mean_vals_[15] = mean_vals[15]

            mean_vals = mean_vals_
        
        print("mean vals ", mean_vals)

        mean_vals_s.append(mean_vals)


    print("len ! :", len(mean_vals_s[0]))
    ep0 = EpisodeData(data_s[0], protocol_name=p, quantities=['dFoF'])

    varied_keys = list(ep0.varied_parameters.keys())
    angles = ep0.varied_parameters[varied_keys[0]]
    contrasts = ep0.varied_parameters[varied_keys[1]]

    #param_values = [f"{a}°\n C={c}" for a in angles for c in contrasts]
    if p == 'ff-gratings-8orientation-2contrasts-15repeats':
        param_values = [f"a={a:.1f}° , C={c:.1f}" for c in contrasts for a in angles]

    elif p == 'ff-gratings-2orientations-8contrasts-15repeats':
        param_values = [f"a={a:.1f}° , C={c:.1f}" for a in angles for c in contrasts]

    # Compute session-aggregated mean and SEM
    print("mean vals s", mean_vals_s)
    values = np.nanmean(mean_vals_s, axis=0)
    yerr = stats.sem(mean_vals_s, axis=0, nan_policy='omit')

    # Plot directly
    x = np.arange(len(values))
    AX.bar(x, values, 
           yerr=yerr,
           alpha=0.8, 
           capsize=0,
           error_kw=dict(linewidth=0.6))
    AX.set_xticks(x)
    AX.set_xticklabels(param_values,rotation=90, ha="right")

    AX.set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    AX.axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX.set_ylabel('variation \ndFoF')
    
    return 0

def generate_figures_GROUP(data_s, subplots_n, test="angle"):
    start_time = time.time()  

    protocols = [p for p in data_s[0].protocols 
                        if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

    fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s, protocol=protocols[0])
    elapsed = time.time() - start_time
    print(f"Fig 1 ok: {elapsed:.2f} seconds")

    fig_vdFoF, AX2  = pt.figure(axes = (len(protocols),1), ax_scale=(2, 1))
    plot_barplot2_of_protocol(data_s, AX2, 0, data_s[0].protocols[0], subplots_n)

    fig_resp_vs_param, AX3 = pt.figure(axes = (1,1),figsize=(2,2), ax_scale=(2, 5))
    if test == "angle":
        final_pos_, final_neg_, final_ns_ = plot_responsiveness2_of_protocol_(data_s, idx, data_s[0].protocols[0], type='ROI', type_ = "contrast")
    elif test== "contrast":
        final_pos_, final_neg_, final_ns_ = plot_responsiveness2_of_protocol_(data_s, idx, data_s[0].protocols[0], type='ROI', type_ = "angle")
    ep = EpisodeData(data_s[0], protocol_name=data_s[0].protocols[0], quantities=['dFoF'])
    x = ep.varied_parameters[test]
    plot_resp_vs_param(x=x, AX=AX3, pos = final_pos_, neg = final_neg_, ns = final_ns_, test=test)

    fig1 = figure_to_array(fig_traces)
    fig2 = figure_to_array(fig_vdFoF)
    fig3 = figure_to_array(fig_resp_vs_param)
    print(f"Fig 2-3 ok: {elapsed:.2f} seconds")

    if test=='contrast':
        return fig1, fig2, fig3

    stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')
    
    response_significance_threshold=0.05
    
    compute_tunings2(data_s = data_s, 
                    stat_test_props=stat_test_props, 
                    response_significance_threshold=response_significance_threshold)
    

    fig_resp_per_param, AX4 = plot_responsiveness(['WT_contrast-1.0', 'WT_contrast-0.5'],
                                    average_by='ROIs', #issue !!  #not average by subject
                                    path=tempfile.tempdir, 
                                    fig_args={'figsize':  (2,1.), 
                                              'ax_scale': (1.,1.)})
    

    fig_selectivity, AX5 = plot_selectivity(['WT_contrast-1.0', 'WT_contrast-0.5'],
                                 average_by='ROIs',
                                 #using='fit',
                                 path=tempfile.tempdir)
    
    fig_tuning_curve, AX6 = plot_orientation_tuning_curve(['WT_contrast-1.0', 'WT_contrast-0.5'],
                                              average_by='ROIs',
                                              path=tempfile.tempdir)
        
    elapsed = time.time() - start_time
    print(f"Fig 4-5-6 ok: {elapsed:.2f} seconds")

    
    fig4 = figure_to_array(fig_resp_per_param)
    fig5 = figure_to_array(fig_selectivity)
    fig6 = figure_to_array(fig_tuning_curve)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")

    if test=='angle':
        return fig1, fig2, fig3, fig4, fig5, fig6
    
def create_group_PDF(fig1, fig2, fig3, fig4=None, fig5=None, fig6=None, cell_type='', test='angle'):
    try: 
        if test=="angle":
            pdf1 = PDF_ori_tuning() #PDF_angle_contrast()
            pdf1.fill_PDF(fig1, fig2, fig3, fig4, fig5, fig6)
        if test=="contrast":
            pdf1 = PDF_angle_contrast()
            pdf1.fill_PDF(fig1, fig2, fig3)

        fig_p1 = pdf1.fig
        output_path = f"C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/GROUP_summary.pdf"
        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1

        print("GROUP PDF File saved successfully ")
    except Exception as e:
        print(f"Error creating GROUP PDF file : {e}")

    return 0

def plot_responsiveness2_of_protocol_(data_s, idx, p, type='means', type_ = "contrast"):

    nROIs = 0

    sig_arrs_s = []
    val_arrs_s = []

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
        
        sig_arrs = [[] for _ in range(8)]
        val_arrs = [[] for _ in range(8)]

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
                                                   repetition_keys=["repeat", type_])
            

            #print(type(roi_summary_data['significant']))

            #if isinstance(bool(roi_summary_data['significant']), np.ndarray):
            for i in range(len(roi_summary_data['significant'])):
                sig_arrs[i].append(bool(roi_summary_data['significant'][i]))
                val_arrs[i].append(roi_summary_data['value'][i])

            #else :
            #    sig_arrs[0].append(bool(roi_summary_data['significant']))
            #    val_arrs[0].append(roi_summary_data['value'])

            
        sig_arrs_s.append(sig_arrs)
        val_arrs_s.append(val_arrs)

        nROIs += data.nROIs
    
    print("here : ", len(sig_arrs_s), len(sig_arrs_s[1]))

    #plot:
    out = []
    out_v = []
    for c in range(len(sig_arrs_s[0])):
        temp_ = []
        temp_v_ = []
        for f in range(len(data_s)):
            temp = sig_arrs_s[f][c]
            temp_v = val_arrs_s[f][c]
            temp_.append(temp)
            temp_v_.append(temp_v)

        flat = [x for sub in temp_ for x in sub]
        out.append(flat)

        flat_v = [x for sub in temp_v_ for x in sub]
        out_v.append(flat_v)



    print(len(out))
    #print(len(out_v), len(out_v[0]), len(out_v[1]))

    final_pos_ = []
    final_neg_ = []
    final_ns_ = []

    for c in range(len(out)):
        data = out[c]
        data_v = out_v[c]
        print(len(data))
        print(len(data_v))

        resp_cond = data
        pos_cond = [data[i] & (data_v[i] > 0) for i in range(len(data_v))]
        neg_cond = [data[i] & (data_v[i] < 0) for i in range(len(data_v))]

        resp_count = np.sum(resp_cond)
        pos_count = np.sum(pos_cond)
        neg_count = np.sum(neg_cond)

        
        print("len resp cond", len(resp_cond))
        final_pos = (pos_count/len(resp_cond))*100
        final_neg = (neg_count/len(resp_cond))*100
        final_ns = ((nROIs - resp_count)/len(resp_cond))*100

        print(final_neg, final_ns)

        final_pos_.append(final_pos)
        final_neg_.append(final_neg)
        final_ns_.append(final_ns)

    return final_pos_, final_neg_, final_ns_

def plot_resp_vs_param(x, AX, pos, neg, ns, test = "angle"):
    
    pt.plot(x, pos, ax=AX, color = "green")
    pt.plot(x, neg, ax=AX, color="red")
    pt.plot(x, ns, ax=AX, color = "grey")

    pt.set_plot(ax = AX, 
                ylabel='Responsiveness (%)', 
                yticks=np.arange(0, 105, 10),
                xlabel=test, 
                fontsize = 15)
    AX.set_xticks(x)
    AX.set_xticklabels([f"{xi:.1f}" for xi in x], rotation=90, ha="right")
    AX.set_ylim(0, 100)
    
    return 0

##################################################################################################################
##################################################################################################################
##################################################################################################################
#%% [markdown]
##################################################################################################################
######################################## 8 ORIENTATIONS 2 contrasts ##############################################
##################################################################################################################
#%% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'NWBs_orientations-test')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}
data_s_ori = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_ori.append(data)

#%% [markdown]
## All individual files
#%%
#generate_figures(data_s, cell_type='NDNF', subplots_n=16)
#%% [mardown]
## GROUPED ANALYSIS

#%%
# 8 orientations 2 contrasts
test = "angle"
fig1, fig2, fig3, fig4, fig5, fig6 = generate_figures_GROUP(data_s_ori, subplots_n=16, test=test)
create_group_PDF(fig1, fig2, fig3, fig4, fig5, fig6, 'NDNF', test=test)

#%% ##############################################################################################################
######################################## 2 ORIENTATIONS 8 contrasts ##############################################
##################################################################################################################
#%% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'NWBs_contrasts-test')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con.append(data)

test = "contrast"

#%% [markdown]
## All individual files
#%%
dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = generate_figures(data_s_con, subplots_n=16)
create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type='NDNF')
#%% [mardown]
## GROUPED ANALYSIS
#%%
fig1, fig2, fig3 = generate_figures_GROUP(data_s_con, subplots_n=16, test=test)
create_group_PDF(fig1, fig2, fig3, 'NDNF', test=test)