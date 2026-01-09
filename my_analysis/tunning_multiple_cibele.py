# %% [markdown]
# # Analyze Temporal Dynamics

# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path.append('../physion/src') # add src code directory for physion

import physion.utils.plot_tools as pt
pt.set_style('manuscript')

summary_folder = os.path.expanduser('~/DATA/summary')
# %% [markdown]
# ## Multiple Session Summary

# %%
def plot_response_dynamics(keys,
                               path=os.path.expanduser('~'),
                               average_by='sessions',
                               norm='',
                               colors=None,
                               with_label=True,
                               fig_args={}):
        
        if colors is None:
            colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

        if type(keys)==str:
            keys, colors = [keys], [colors[0]]

        fig, ax = pt.figure(**fig_args)
        x = np.linspace(-30, 180-30, 100)

        for i, (key, color) in enumerate(zip(keys, colors)):

                # load data
                Responses = \
                        np.load(os.path.join(path, 'Deconvolved_%s.npy' % key), 
                                allow_pickle=True)
                
                if len(Responses)>0:

                        if average_by=='sessions':
                                # mean significant responses per session
                                Deconvolved = [np.mean(Response['Deconvolved'][Response['significant'],:],
                                                axis=0) for Response in Responses]

                        elif average_by=='ROIs':
                                # mean significant responses per session
                                Deconvolved = np.concatenate([\
                                                Response['Deconvolved'][Response['significant'],:]\
                                                                        for Response in Responses])

                        if norm == 'min-max':
                                response = np.mean(Deconvolved, axis=0)
                                sresponse = stats.sem(Deconvolved, axis=0)
                                response -= response[Responses[0]['t']<0].mean()
                                sresponse /= response.max() # first sem
                                response /= response.max()
                                pt.plot(Responses[0]['t'], 
                                        response, sy=sresponse,
                                        color=color, ax=ax, ms=2)
                        
                        else:
                                pt.plot(Responses[0]['t'], 
                                        np.mean(Deconvolved, axis=0), 
                                                sy=stats.sem(Deconvolved, axis=0), 
                                                color=color, ax=ax, ms=2)

                        if with_label:

                                annot = i*'\n'
                                if average_by=='sessions':
                                        annot += 'N=%02d %s, ' % (len(Deconvolved), average_by) + key
                                else:
                                        annot += 'n=%04d %s, ' % (len(Deconvolved), average_by) + key

                                pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

        pt.set_plot(ax, ylabel='(%s)\n$\Delta$F/F' % norm,  xlabel='time (s)')

        return fig, ax
    
#DATASETS = ['PYR-PV-SynGCaMP_WT_P15-P19_V1_contrast-1.0',
#            'PYR-PV-SynGCaMP_WT_P20-P23_V1_contrast-1.0',
#            'PYR-PV-SynGCaMP_WT_P24-P27_V1_contrast-1.0']

DATASETS = [os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch2', 'NWBs')]

fig, ax = plot_response_dynamics(DATASETS[0],\
                        # average_by='ROIs',
                        norm='min-max')
                        
# 

# %%f
def plot_response_dynamics(keys,
                               path=os.path.expanduser('~'),
                               average_by='sessions',
                               norm='',
                               colors=None,
                               with_label=True,
                               fig_args={}):
        
        if colors is None:
            colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

        if type(keys)==str:
            keys, colors = [keys], [colors[0]]

        fig, ax = pt.figure(**fig_args)
        x = np.linspace(-30, 180-30, 100)

        for i, (key, color) in enumerate(zip(keys, colors)):

                # load data
                Responses = np.load(
                    os.path.join(path, f'Deconvolved_{key}.npy'),
                    allow_pickle=True
                )

                if len(Responses) > 0:

                        # ------------------------------------------
                        # FIXED SECTION: fallback when no significant ROIs
                        # ------------------------------------------
                        if average_by == 'sessions':
                            Deconvolved = []
                            for Response in Responses:
                                sig = Response['significant']
                                # fallback to all ROIs if none significant
                                if sig.sum() == 0:
                                    d = Response['Deconvolved']
                                else:
                                    d = Response['Deconvolved'][sig]
                                Deconvolved.append(np.mean(d, axis=0))

                        elif average_by == 'ROIs':
                            Deconvolved = []
                            for Response in Responses:
                                sig = Response['significant']
                                if sig.sum() == 0:
                                    d = Response['Deconvolved']
                                else:
                                    d = Response['Deconvolved'][sig]
                                Deconvolved.append(d)
                            Deconvolved = np.concatenate(Deconvolved)
                        # ------------------------------------------

                        # Normalization / plotting
                        if norm == 'min-max':
                                response = np.mean(Deconvolved, axis=0)
                                sresponse = stats.sem(Deconvolved, axis=0)

                                # subtract pre-stimulus baseline
                                response -= response[Responses[0]['t']<0].mean()

                                # scale
                                sresponse /= response.max()
                                response /= response.max()

                                pt.plot(Responses[0]['t'], 
                                        response, sy=sresponse,
                                        color=color, ax=ax, ms=2)
                        
                        else:
                                pt.plot(Responses[0]['t'], 
                                        np.mean(Deconvolved, axis=0), 
                                        sy=stats.sem(Deconvolved, axis=0), 
                                        color=color, ax=ax, ms=2)

                        if with_label:

                                annot = i*'\n'
                                if average_by == 'sessions':
                                        annot += 'N=%02d %s, ' % (len(Deconvolved), average_by) + key
                                else:
                                        annot += 'n=%04d %s, ' % (len(Deconvolved), average_by) + key

                                pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

        pt.set_plot(ax, ylabel='(%s)\n$\Delta$F/F' % norm, xlabel='time (s)')

        return fig, ax

#%%
DATASETS = [os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch2', 'NWBs')]

fig, ax = plot_response_dynamics(DATASETS,\
                        # average_by='ROIs',
                        norm='min-max',
                        colors= ['tan', 'burlywood', 'orange'],
                        path=summary_folder)












#%% 
# # Different plots 
#%%
import matplotlib.pyplot as plt
plt.style.use('default')   # impede eixos cinzas

def plot_response_dynamics(keys,
                           path=os.path.expanduser('~'),
                           average_by='sessions',
                           norm='',
                           colors=None,
                           with_label=True,
                           fig_args={},
                           zoom_window=[0, 0.5],
                           add_zoom=True):

    if colors is None:
        colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type(keys)==str:
        keys, colors = [keys], [colors[0]]

    # -----------------------------------------
    # Physion figure
    # -----------------------------------------
    fig, ax = pt.figure(**fig_args)

    # inset
    zoom_ax = pt.inset(ax, [0.55, 0.45, 0.35, 0.45]) if add_zoom else None

    # -----------------------------------------
    # loop datasets
    # -----------------------------------------
    for i, (key, color) in enumerate(zip(keys, colors)):

        Responses = np.load(
            os.path.join(path, f'Deconvolved_{key}.npy'),
            allow_pickle=True
        )
        if len(Responses) == 0:
            continue

        # choose rois
        if average_by == 'sessions':
            Deconvolved = []
            for Response in Responses:
                sig = Response['significant']
                d = Response['Deconvolved'] if sig.sum()==0 else Response['Deconvolved'][sig]
                Deconvolved.append(np.mean(d, axis=0))
        else:
            Deconvolved = []
            for Response in Responses:
                sig = Response['significant']
                d = Response['Deconvolved'] if sig.sum()==0 else Response['Deconvolved'][sig]
                Deconvolved.append(d)
            Deconvolved = np.concatenate(Deconvolved)

        t = Responses[0]['t']

        resp = np.mean(Deconvolved, axis=0)
        sresp = stats.sem(Deconvolved, axis=0)

        if norm == 'min-max':
            resp = resp - resp[t<0].mean()
            resp /= resp.max()
            sresp /= resp.max()

        pt.plot(t, resp, sy=sresp, color=color, ax=ax, ms=2)

        if add_zoom:
            mask = (t>=zoom_window[0]) & (t<=zoom_window[1])
            pt.plot(t[mask], resp[mask], sy=sresp[mask], color=color, ax=zoom_ax, ms=2)

        # label
        if with_label:
            annot = i*'\n'
            if average_by == 'sessions':
                annot += f"N={len(Deconvolved):02d} {average_by}, {key}"
            else:
                annot += f"n={len(Deconvolved):04d} {average_by}, {key}"
            pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

    # ======================================================
    # FORCE BLACK AXES (agora dentro da função!)
    # ======================================================
    targets = [ax]
    if add_zoom:
        targets.append(zoom_ax)

    for axis in targets:

        axis.set_facecolor('white')

        for spine in axis.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        axis.tick_params(axis='both',
                         which='both',
                         color='black',
                         labelcolor='black')

        axis.xaxis.label.set_color('black')
        axis.yaxis.label.set_color('black')

    # Apply physion formatting
    pt.set_plot(ax, ylabel=f'({norm})\nΔF/F', xlabel='time (s)')

    if add_zoom:
        pt.set_plot(zoom_ax, ['left', 'bottom'])

    return fig, ax


DATASETS = ['SST-cells_WT_P20-P23_V1_contrast-1.0',
            'SST-cells_WT_P24-P27_V1_contrast-1.0',
            'SST-cells_WT_Adult_V1_contrast-1.0']


fig, ax = plot_response_dynamics(DATASETS,\
                        # average_by='ROIs',
                        norm='min-max',
                        colors= ['tab:brown', 'peru', 'tab:orange'],
                        path=summary_folder)


#---SAVE FIGURE
#fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Temporal-dynamics_SST-cells_WT_Young-vs-Adult.svg'))


#%%
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import physion.utils.plot_tools as pt


# ============================================================
#   1) FUNCTION TO COMPUTE PEAK RESPONSE ("Neuron-style")
# ============================================================
def compute_peak_response(Deconvolved, t,
                          baseline_window=(-1, 0),
                          response_window=(0, 1)):
    """
    Computes peak ΔF/F after subtracting pre-stim baseline.
    Returns peak per-ROI.
    """
    bmask = (t >= baseline_window[0]) & (t < baseline_window[1])
    rmask = (t >= response_window[0]) & (t <= response_window[1])

    peaks = []
    for r in Deconvolved:
        baseline = r[bmask].mean()
        resp = r - baseline
        peak = resp[rmask].max()
        peaks.append(peak)

    return np.array(peaks)


# ============================================================
#   2) PLOTTING FUNCTION WITH ZOOM + PEAK BARPLOT
# ============================================================
def plot_response_dynamics_and_peaks(
        keys,
        path,
        colors,
        average_by="ROIs",
        zoom_window=(0, 0.5),
        norm="min-max",
        add_zoom=True):

    n_groups = len(keys)

    # MAIN FIGURE WITH SUBPLOTS
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax_bar = fig.add_subplot(2, 1, 2)

    # Collect peaks for stats
    all_peaks = []

    #===========================================
    # LOOP THROUGH GROUPS
    #===========================================
    for i, (key, color) in enumerate(zip(keys, colors)):

        # Load data
        Responses = np.load(
            os.path.join(path, f"Deconvolved_{key}.npy"),
            allow_pickle=True
        )
        t = Responses[0]['t']

        # Build ROI matrix
        Deconvolved = []
        for R in Responses:
            sig = R['significant']
            d = R['Deconvolved'] if sig.sum() == 0 else R['Deconvolved'][sig]
            Deconvolved.append(d)
        Deconvolved = np.concatenate(Deconvolved)

        # Mean response
        resp = np.mean(Deconvolved, axis=0)
        sresp = stats.sem(Deconvolved, axis=0)

        # Normalize
        if norm == "min-max":
            resp = resp - resp[t < 0].mean()
            resp /= resp.max()
            sresp /= resp.max()

        # Plot trace
        pt.plot(t, resp, sy=sresp, color=color, ax=ax, ms=2)

        # Zoom inset
        if add_zoom:
            mask = (t >= zoom_window[0]) & (t <= zoom_window[1])
            if i == 0:   # create only once
                zoom_ax = pt.inset(ax, [0.55, 0.45, 0.35, 0.45])
            pt.plot(t[mask], resp[mask], sy=sresp[mask],
                    color=color, ax=zoom_ax, ms=2)

        # Compute peak ("Neuron-style")
        peaks = compute_peak_response(Deconvolved, t,
                                      baseline_window=(-1, 0),
                                      response_window=(0, 1))

        all_peaks.append(peaks)

        # Bar plot
        ax_bar.bar(i, peaks.mean(), yerr=stats.sem(peaks),
                   color=color, alpha=0.8, width=0.6)
        ax_bar.scatter(np.repeat(i, len(peaks)), peaks,
                       color='black', s=10, alpha=0.5)

    #===========================================
    # STYLING
    #===========================================
    ax.set_title("Temporal Dynamics (mean ± SEM)", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (normalized)")

    ax_bar.set_title("Peak Response (0–1 s window)", fontsize=12)
    ax_bar.set_ylabel("Peak ΔF/F")
    ax_bar.set_xticks(range(n_groups))
    ax_bar.set_xticklabels(keys, rotation=45, ha='right')

    fig.tight_layout()

    return fig, ax, ax_bar, all_peaks


# ============================================================
# 3) RUN ANALYSIS EXAMPLE
# ============================================================

DATASETS = [
    "SST-cells_WT_P20-P23_V1_contrast-1.0",
    "SST-cells_WT_P24-P27_V1_contrast-1.0",
    "SST-cells_WT_Adult_V1_contrast-1.0"
]

COLORS = ["tab:brown", "peru", "tab:orange"]

summary_folder = os.path.expanduser("~/DATA/summary")

fig, ax, ax_bar, all_peaks = plot_response_dynamics_and_peaks(
    keys=DATASETS,
    path=summary_folder,
    colors=COLORS
)

#fig.savefig(os.path.expanduser("~/Desktop/Neuron-style_analysis.svg"))
print("FIGURE SAVED.")

# ============================================================
# 4) STATISTICS (like Neuron)
# ============================================================

# ANOVA across groups
fval, pval = stats.f_oneway(*all_peaks)
print("\n=== ANOVA ===")
print("F =", fval, "p =", pval)

# Pairwise t-tests
print("\n=== Pairwise t-tests ===")
for i in range(len(all_peaks)):
    for j in range(i+1, len(all_peaks)):
        t, p = stats.ttest_ind(all_peaks[i], all_peaks[j])
        print(f"{DATASETS[i]}  vs  {DATASETS[j]}: p = {p}")









#%%
#%%
import numpy as np
import os
from scipy import stats
import itertools
import matplotlib.pyplot as plt
import physion.utils.plot_tools as pt

def compute_peak_response(Deconvolved, t, baseline_window=(-1, 0), response_window=(0, 1)):
    """
    NEURON-style peak per ROI:
      - baseline = mean in baseline_window
      - response = trace - baseline
      - peak = max(response) in response_window
    Deconvolved: (n_rois, n_timepoints)
    Returns: 1D array of peak values (per ROI)
    """
    bmask = (t >= baseline_window[0]) & (t < baseline_window[1])
    rmask = (t >= response_window[0]) & (t <= response_window[1])
    peaks = []
    for r in Deconvolved:
        baseline = np.mean(r[bmask])
        resp = r - baseline
        peak = np.max(resp[rmask])
        peaks.append(peak)
    return np.array(peaks)


def _format_p(p):
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    if p >= 0.0005:
        pstr = f"p={p:.3f}"
    else:
        pstr = f"p={p:.1e}"
    return stars, pstr


def plot_response_dynamics_and_peaks_with_stats(
        keys,
        path,
        colors,
        average_by="ROIs",
        zoom_window=(0, 0.5),
        norm="min-max",
        add_zoom=True,
        figsize=(10, 8),
        savepath=None):

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 0.05, 1], hspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[2, 0])

    all_peaks = []
    labels_short = []
    zoom_ax = None
    n_groups = len(keys)

    # ====================================================
    # MAIN LOOP
    # ====================================================
    for i, (key, color) in enumerate(zip(keys, colors)):

        Responses = np.load(os.path.join(path, f"Deconvolved_{key}.npy"), allow_pickle=True)
        if len(Responses) == 0:
            print(f"[warning] no responses for {key}")
            all_peaks.append(np.array([]))
            continue

        t = Responses[0]['t']

        # Collect all significant ROIs
        Deconvolved_list = []
        for R in Responses:
            sig = R.get('significant', np.ones(R["Deconvolved"].shape[0], dtype=bool))
            if sig.sum() == 0:
                d = R["Deconvolved"]
            else:
                d = R["Deconvolved"][sig]
            Deconvolved_list.append(d)

        Deconvolved = np.concatenate(Deconvolved_list, axis=0)

        # ====================================================
        # >>> ADDED: N sessions, n ROIs, N animals (subject)
        # ====================================================
        N_sessions = len(Responses)
        n_rois = Deconvolved.shape[0]

        animal_ids = []
        for R in Responses:
            if "subject" in R:
                animal_ids.append(R["subject"])
            else:
                animal_ids.append(None)

        if all(a is None for a in animal_ids):
            N_animals = "?"
            print(f"[INFO] {key}: N={N_sessions} sessions, n={n_rois} ROIs, N_animals=? (no 'subject' field)")
        else:
            ids_clean = [a for a in animal_ids if a is not None]
            N_animals = len(set(ids_clean))
            print(f"[INFO] {key}: N={N_sessions} sessions, n={n_rois} ROIs, N_animals={N_animals}")
        # ====================================================

        # Mean ± SEM traces
        mean_trace = Deconvolved.mean(axis=0)
        sem_trace = stats.sem(Deconvolved, axis=0)

        # Normalization
        if norm == "min-max":
            baseline_mask = (t < 0)
            base = mean_trace[baseline_mask].mean()
            mean_trace = mean_trace - base
            denom = mean_trace.max() if mean_trace.max() != 0 else 1
            mean_trace = mean_trace / denom
            sem_trace = sem_trace / denom

        pt.plot(t, mean_trace, sy=sem_trace, color=color, ax=ax, ms=2)

        if add_zoom:
            if zoom_ax is None:
                zoom_ax = pt.inset(ax, [0.60, 0.55, 0.35, 0.35])
            mask = (t >= zoom_window[0]) & (t <= zoom_window[1])
            pt.plot(t[mask], mean_trace[mask], sy=sem_trace[mask], color=color, ax=zoom_ax, ms=2)

        # Peaks
        peaks = compute_peak_response(Deconvolved, t)
        all_peaks.append(peaks)

        # Bar + scatter
        ax_bar.bar(i, peaks.mean(), yerr=stats.sem(peaks), color=color, alpha=0.9,
                   width=0.6, edgecolor='black')
        ax_bar.scatter(np.full_like(peaks, i) + (np.random.randn(len(peaks)) * 0.05),
                       peaks, color='k', s=8, alpha=0.6)

        labels_short.append(key)

    # Style
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔF/F (normalized)" if norm == "min-max" else "ΔF/F")
    ax.set_title("Temporal dynamics (mean ± SEM)")

    if zoom_ax is not None:
        pt.set_plot(zoom_ax, ['left', 'bottom'])
        zoom_ax.set_title("zoom", fontsize=9)

    ax_bar.set_xticks(range(n_groups))
    ax_bar.set_xticklabels(labels_short, rotation=45, ha='right')
    ax_bar.set_ylabel("Peak ΔF/F (0–1 s)")
    ax_bar.set_title("Peak responses")

    # Axis styling
    for axis in [ax, ax_bar] + ([zoom_ax] if zoom_ax is not None else []):
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
        axis.tick_params(color="black", labelcolor="black")
        axis.xaxis.label.set_color("black")
        axis.yaxis.label.set_color("black")

    # ====================================================
    # STATISTICS: pairwise t-tests
    # ====================================================
    pair_indices = list(itertools.combinations(range(n_groups), 2))
    pair_results = []

    for (i, j) in pair_indices:
        a = all_peaks[i]
        b = all_peaks[j]
        if a.size == 0 or b.size == 0:
            tval, pval = np.nan, np.nan
        else:
            tval, pval = stats.ttest_ind(a, b, equal_var=False)
        pair_results.append((i, j, tval, pval))

    # Brackets
    bar_tops = []
    for peaks in all_peaks:
        if peaks.size == 0:
            bar_tops.append(0)
        else:
            bar_tops.append(peaks.mean() + stats.sem(peaks))

    data_top = np.max(bar_tops)
    max_point = np.max([p.max() if p.size > 0 else -np.inf for p in all_peaks])
    baseline_y = max(data_top, max_point)
    offset = (abs(baseline_y) + 1e-6) * 0.08
    if offset == 0: offset = 0.1

    current_heights = []
    for (i, j, tval, pval) in sorted(pair_results,
                                    key=lambda x: (x[1] - x[0], x[0])):
        y = baseline_y + offset * (1 + len(current_heights) * 0.5)
        while any(abs(y - yh) < offset * 0.9 for yh in current_heights):
            y += offset * 0.6
        current_heights.append(y)

        h = offset * 0.3
        ax_bar.plot([i, i, j, j], [y, y + h, y + h, y], lw=1.0, color='black')

        stars, pstr = _format_p(pval if not np.isnan(pval) else 1)
        txt = f"{stars} ({pstr})" if stars != "n.s." else f"n.s. ({pstr})"
        ax_bar.text((i + j) / 2, y + h + offset * 0.05, txt,
                    ha='center', va='bottom', fontsize=9, color='black')

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, facecolor="white", bbox_inches="tight")

    print("\nPairwise comparisons (Welch t-test):")
    for (i, j, tval, pval) in pair_results:
        print(f"{keys[i]} vs {keys[j]}: t={tval:.3f}, p={pval}")

    return fig, ax, ax_bar, all_peaks, pair_results


# ====================================================
# RUN FUNCTION
# ====================================================

DATASETS = [
    "PV-cells_WT_Adult_V1_contrast-1.0",
    "PV-cells_cond-GluN1-KO_Adult_V1_contrast-1.0",
]

COLORS = ["tab:red", "tab:green"]
summary_folder = os.path.expanduser("~/DATA/summary")

fig, ax, ax_bar, all_peaks, pair_results = plot_response_dynamics_and_peaks_with_stats(
    keys=DATASETS,
    path=summary_folder,
    colors=COLORS,
    average_by="ROIs",
    zoom_window=(0, 0.4),
    norm="min-max",
    add_zoom=True,
    savepath=os.path.expanduser("~/Desktop/Neuron_style_peaks.svg")
)

#---SAVE FIGURE
#fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Temporal-dynamics_peak-response_WT_Young-P24-27.svg'))




















#%%
# TEST
## Statistics on late-window mean responses across multiple sessions
import os
import numpy as np
from scipy import stats

# ----------------------------
# User parameters
# ----------------------------
DATASETS = [
    "SST-cells_WT_P15-P19_V1_contrast-1.0",
    "SST-cells_WT_P20-P23_V1_contrast-1.0",
    "SST-cells_WT_P24-P27_V1_contrast-1.0",
    "SST-cells_WT_Adult_V1_contrast-1.0",
]

path = os.path.expanduser("~/DATA/summary")
late_window = (0.8, 1.8)

# ----------------------------
# Extract late-window mean per ROI
# ----------------------------
print("\n===============================")
print(f" Late-window comparison {late_window[0]}–{late_window[1]} s")
print("===============================\n")

late_values = {}

for key in DATASETS:

    filename = os.path.join(path, f"Deconvolved_{key}.npy")
    if not os.path.isfile(filename):
        print(f"[WARNING] File not found: {filename}")
        late_values[key] = np.array([])
        continue

    Responses = np.load(filename, allow_pickle=True)
    if len(Responses) == 0:
        print(f"[WARNING] No responses inside {filename}")
        late_values[key] = np.array([])
        continue

    # read time vector from first response
    t = Responses[0]["t"]
    mask_late = (t >= late_window[0]) & (t <= late_window[1])

    vals = []

    for R in Responses:
        sig = R.get("significant", np.ones(R["Deconvolved"].shape[0], dtype=bool))
        d = R["Deconvolved"][sig] if sig.sum() > 0 else R["Deconvolved"]

        # mean per ROI inside late window
        vals.append(d[:, mask_late].mean(axis=1))

    late_values[key] = np.concatenate(vals)

# ----------------------------
# Print values summary
# ----------------------------
for key in DATASETS:
    arr = late_values[key]
    print(f"{key}: n={len(arr)}, mean={arr.mean():.4f}, sem={stats.sem(arr) if len(arr)>1 else np.nan:.4f}")

# ----------------------------
# Pairwise statistics
# ----------------------------
print("\nPairwise Welch t-tests:\n")

for i in range(len(DATASETS)):
    for j in range(i+1, len(DATASETS)):

        a = late_values[DATASETS[i]]
        b = late_values[DATASETS[j]]

        if len(a)==0 or len(b)==0:
            print(f"{DATASETS[i]} vs {DATASETS[j]}: missing data")
            continue

        tval, pval = stats.ttest_ind(a, b, equal_var=False)

        print(f"{DATASETS[i]}  vs  {DATASETS[j]}:  "
              f"t={tval:.3f}, p={pval:.3g}")










#%%
# TEST
## Statistics + bar plot with p-value annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# User parameters
# ----------------------------
DATASETS = [
    "PV-cells_WT_Adult_V1_contrast-1.0",
    "PV-cells_cond-GluN1-KO_Adult_V1_contrast-1.0",
]

path = os.path.expanduser("~/DATA/summary")
late_window = (0.8, 1.8)

# ----------------------------
# Extract late-window mean per ROI
# ----------------------------
print("\n===============================")
print(f" Late-window comparison {late_window[0]}–{late_window[1]} s")
print("===============================\n")

late_values = {}

for key in DATASETS:

    filename = os.path.join(path, f"Deconvolved_{key}.npy")
    if not os.path.isfile(filename):
        print(f"[WARNING] File not found: {filename}")
        late_values[key] = np.array([])
        continue

    Responses = np.load(filename, allow_pickle=True)
    if len(Responses) == 0:
        print(f"[WARNING] No responses inside {filename}")
        late_values[key] = np.array([])
        continue

    t = Responses[0]["t"]
    mask_late = (t >= late_window[0]) & (t <= late_window[1])

    vals = []
    for R in Responses:
        sig = R.get("significant", np.ones(R["Deconvolved"].shape[0], dtype=bool))
        d = R["Deconvolved"][sig] if sig.sum() > 0 else R["Deconvolved"]
        vals.append(d[:, mask_late].mean(axis=1))

    late_values[key] = np.concatenate(vals)

# ----------------------------
# Print summary
# ----------------------------
for key in DATASETS:
    arr = late_values[key]
    print(f"{key}: n={len(arr)}, mean={arr.mean():.4f}, "
          f"sem={stats.sem(arr) if len(arr)>1 else np.nan:.4f}")

# ----------------------------
# Bar plot + p-value annotations
# ----------------------------
def plot_bar_with_pvalues(late_values, datasets, ylabel, title):

    means = []
    sems = []

    for k in datasets:
        arr = late_values[k]
        means.append(arr.mean() if len(arr) > 0 else np.nan)
        sems.append(stats.sem(arr) if len(arr) > 1 else np.nan)

    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        x, means, yerr=sems,
        capsize=5, edgecolor="black", alpha=0.8
    )

    # Scatter individual ROI points
    for i, k in enumerate(datasets):
        y = late_values[k]
        jitter = np.random.normal(i, 0.05, size=len(y))
        ax.plot(jitter, y, 'k.', alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------- p-value annotations ----------
    ymax = max(means) + max(sems) * 2
    y_step = ymax * 0.08
    current_y = ymax

    def p_to_star(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "n.s."

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a = late_values[datasets[i]]
            b = late_values[datasets[j]]

            if len(a) == 0 or len(b) == 0:
                continue

            tval, pval = stats.ttest_ind(a, b, equal_var=False)
            label = p_to_star(pval)

            ax.plot([i, i, j, j],
                    [current_y, current_y + y_step,
                     current_y + y_step, current_y],
                    lw=1.2, c="black")

            ax.text((i + j) / 2, current_y + y_step,
                    label, ha="center", va="bottom")

            current_y += y_step * 1.2

    ax.set_ylim(0, current_y + y_step)
    plt.tight_layout()
    return fig, ax

# ----------------------------
# Make the plot
# ----------------------------
fig, ax = plot_bar_with_pvalues(
    late_values,
    DATASETS,
    ylabel="Late-window mean response (a.u.)",
    title=f"Late-window responses ({late_window[0]}–{late_window[1]} s)"
)

plt.show()








# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pynwb import NWBHDF5IO

# ------------------------------------------------------------
# Load NWB file
# ------------------------------------------------------------
filename = os.path.join(os.path.expanduser('~'),
                        'CURATED', 'Cibele', 'PYR-PV-SynGCaMP_WT_Young_V1', 'NWBs',
                        '2025_06_04-10-15-37.nwb')

io = NWBHDF5IO(filename, 'r')
nwb = io.read()

ophys = nwb.processing["ophys"]

# ------------------------------------------------------------
# Load mean image (campo de visão)
# ------------------------------------------------------------
img = ophys.data_interfaces["Backgrounds_0"].images["meanImg"].data[:]
low, high = np.percentile(img, 2), np.percentile(img, 98)
img_disp = np.clip((img - low) / (high - low), 0, 1)

# ------------------------------------------------------------
# Load ΔF/F
# ------------------------------------------------------------
flu = ophys.data_interfaces["Fluorescence"]
rrs = list(flu.roi_response_series.values())[0]

dff = rrs.data[:]              # shape (T, N)
time = rrs.timestamps[:]       # timestamps
T, N = dff.shape

# ------------------------------------------------------------
# Detect 40s window with highest activity
# ------------------------------------------------------------
activity = np.mean(np.abs(dff), axis=1)
dt = np.median(np.diff(time))

win40 = int(40 / dt)
conv = np.convolve(activity, np.ones(win40), mode='valid')
best_idx = np.argmax(conv)

t0_40 = time[best_idx]
t1_40 = time[min(best_idx + win40, T - 1)]
mask40 = (time >= t0_40) & (time <= t1_40)

# ------------------------------------------------------------
# Select inner 5s segment
# ------------------------------------------------------------
win5 = int(5 / dt)
center5 = best_idx + win40 // 2

start5 = max(center5 - win5 // 2, 0)
end5 = min(start5 + win5, T)

t0_5 = time[start5]
t1_5 = time[end5 - 1]
mask5 = (time >= t0_5) & (time <= t1_5)

# ------------------------------------------------------------
# Select ROIs for plotting
# ------------------------------------------------------------
example_rois = [0]              # ROI shown in 5s zoom

rng = np.random.default_rng(0)
bottom_rois = rng.choice(N, 8, replace=False)

# ------------------------------------------------------------
# Create Figure (meanImg + traces)
# ------------------------------------------------------------
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'

fig = plt.figure(figsize=(15, 8), facecolor='white')
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3], height_ratios=[1, 1.4],
                      wspace=0.35, hspace=0.4)

ax_img   = fig.add_subplot(gs[:, 0])
ax_zoom  = fig.add_subplot(gs[0, 1:])
ax_stack = fig.add_subplot(gs[1, 1:])

# ------------------------------------------------------------
# 1) Show anatomical field of view (meanImg)
# ------------------------------------------------------------
ax_img.imshow(img_disp, cmap='gray')
ax_img.set_title("Field of view (mean image)")
ax_img.axis('off')

# ------------------------------------------------------------
# 2) 5-second zoom
# ------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, len(example_rois)))

for i, roi in enumerate(example_rois):
    ax_zoom.plot(time[mask5], dff[mask5, roi], lw=2, color=colors[i])

ax_zoom.set_title("5-second zoom window")
ax_zoom.set_ylabel("ΔF/F")

# ------------------------------------------------------------
# 3) 40-second panel with 8 ROIs
# ------------------------------------------------------------
offset = 3 * np.nanstd(dff)

for i, roi in enumerate(bottom_rois):
    ax_stack.plot(time[mask40], dff[mask40, roi] + i * offset, lw=1.1)

zoom_width = t1_5 - t0_5

rect = Rectangle((t0_5, -offset),
                 zoom_width,
                 offset * (len(bottom_rois) + 1),
                 linewidth=2, edgecolor='yellow', facecolor='none')

ax_stack.add_patch(rect)

ax_stack.set_title("8 ROIs – 40-s window (yellow = 5-s zoom segment)")
ax_stack.set_xlabel("Time (s)")
ax_stack.set_yticks([])

plt.tight_layout()

def save_svg(fig, filepath):
    """
    Salva uma figura Matplotlib em SVG com fundo branco.
    """
    fig.savefig(
        filepath,
        format='svg',
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"SVG salvo em:\n{filepath}")

#save_svg(
    #fig,
   # os.path.expanduser('~/Desktop/Final-Figures-PhD/traces-exemple-PYR-P=25.svg')
#)


plt.show()
io.close()

# %%
