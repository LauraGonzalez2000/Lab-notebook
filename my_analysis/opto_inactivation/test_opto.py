# %%
import os
import sys
import random
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# physion (path resolved from this file, so it works whatever the working directory)
try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive window without __file__
    _HERE = Path.cwd()
sys.path += [str(_HERE.parents[1] / 'physion' / 'src'),  # physion source
             str(_HERE.parent)]                          # my_analysis (for utils_)

from physion.utils import plot_tools as pt
from physion.dataviz.imaging import get_FOV_image
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.read_NWB import Data

from utils_.Responsiveness_methods import calc_responsiveness2
from utils_.my_math import calc_stats, plot_stats

# pt.set_style('dark')
pt.set_style('manuscript')
cmap = pt.get_linear_colormap('lightgreen', 'darkgreen')

VERBOSE = True

# Ep attribute for each trace type
QUANTITY_ATTR = {'dFoF': 'dFoF',
                 'raw_fluo': 'rawFluo',
                 'neuropil': 'neuropil',
                 'Zscore_correctedFluo': 'Zscore_correctedFluo'}

TRACE_LABELS = {'dFoF': 'dFoF',
                'raw_fluo': 'raw ROI fluo',
                'Zscore_correctedFluo': 'Zscore(correctedFluo)'}

VARIATION_YLABELS = {'dFoF': r'$\Delta$ $\Delta$F/F',
                     'Zscore_correctedFluo': 'Zscore\n(corrFluo)'}

# LED color per setup
SETUP_COLORS = {'stg': 'lightskyblue',   # 450-455 nm (blue)
                'archt': 'greenyellow'}  # 590 nm (orange)
DEFAULT_LED_COLOR = 'lightskyblue'

QUANTITIES = ['dFoF', 'rawFluo', 'neuropil', 'opto', 'Zscore_correctedFluo']

# %%
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
DRIFTING_GRATING_NAME = 'protocol-ff_drifting_grating'

#%%

def fix_protocol_names(data, verbose=False):
    """
    Older vision-survey files have the drifting grating (protocol index 1)
    wrongly named 'protocol-natural-images', so that name appears twice.
    Renames the first occurrence to the name used in fixed recordings,
    so protocols can be selected by name in both old and new files.
    Call it after data.build_visual_stim().
    """
    names = list(data.protocols)
    if names.count('protocol-natural-images') == 2:
        i = names.index('protocol-natural-images')
        names[i] = DRIFTING_GRATING_NAME
        # rebuild the array: the original fixed-width dtype would truncate the longer name
        data.protocols = np.array(names, dtype=str)
        data.metadata['protocols'] = data.protocols
        if verbose:
            print('[fix_protocol_names] protocol %i renamed to %s' % (i, DRIFTING_GRATING_NAME))
    return data


def get_protocol_id(data, protocol_name):
    names = list(data.protocols)
    if protocol_name not in names:
        raise ValueError('%r not found in protocols %s' % (protocol_name, names))
    return names.index(protocol_name)


def get_trace(Ep, quantity, roi=None):
    """
    roi can be None (average all rois),
        a specific roi (int),
        or a set of rois (average of those rois)
    returns an array of shape (n_episodes, n_timepoints)
    """
    if quantity not in QUANTITY_ATTR:
        raise ValueError('unknown quantity %r, choose from %s' % (quantity, list(QUANTITY_ATTR)))
    arr = getattr(Ep, QUANTITY_ATTR[quantity])
    if roi is None:
        return arr.mean(axis=1)
    if np.isscalar(roi):
        return arr[:, roi, :]
    return arr[:, roi, :].mean(axis=1)


def make_title(data, protocol_name, rois_desc):
    return '%s,\n%s\n %s' % (data.filename, rois_desc, protocol_name)


def find_EP_LED_on_off(Ep,
                       opto_threshold=0.05,
                       pre_window=0.5,
                       min_fraction=0.90,
                       verbose=False):
    """
    LED_on : LED above threshold for >= min_fraction of
             [-pre_window, 0] AND of the stimulus duration
    LED_off: LED below threshold for >= min_fraction of the same windows
    (min_fraction < 1 tolerates small jitter at the edges)
    """
    pre_mask = (Ep.t >= -pre_window) & (Ep.t < 0.0)
    dur_mask = (Ep.t >= 0.0) & (Ep.t <= Ep.time_duration[0])

    def is_state(above):
        sig = Ep.opto > opto_threshold if above else Ep.opto < opto_threshold
        return (np.mean(sig[:, pre_mask], axis=1) >= min_fraction) & \
               (np.mean(sig[:, dur_mask], axis=1) >= min_fraction)

    LED_on, LED_off = is_state(True), is_state(False)

    if verbose:
        n = len(LED_on)
        print('# LED on episodes', np.sum(LED_on), '/', n)
        print('# LED off episodes', np.sum(LED_off), '/', n)
        n_excl = np.sum(~LED_on & ~LED_off)
        if n_excl:
            print('# episodes neither on nor off (excluded)', n_excl, '/', n)

    return LED_on, LED_off


def _check_LED_masks(Ep, LED_masks, verbose=False):
    if LED_masks is None:
        return find_EP_LED_on_off(Ep, verbose=verbose)
    LED_on, LED_off = LED_masks
    n_ep = Ep.opto.shape[0]
    if len(LED_on) != n_ep or len(LED_off) != n_ep:
        raise ValueError('LED masks (%i, %i) do not match number of episodes (%i)'
                         % (len(LED_on), len(LED_off), n_ep))
    return LED_on, LED_off


def _condition_grid(Ep, average_by):
    """
    Returns grid[row][col] = (episode_condition, label)
    rows: values of the 2nd varied parameter, cols: values of the 1st
    """
    keys = list(Ep.varied_parameters.keys())
    all_ep = np.ones(Ep.opto.shape[0], dtype=bool)

    if average_by == 'stimulus' or len(keys) == 1:
        return [[(all_ep, '')]]

    if len(keys) == 2:
        k0 = keys[0]
        return [[(Ep.find_episode_cond(k0, value=v0), '%s=%.2f' % (k0, v0))
                 for v0 in Ep.varied_parameters[k0]]]

    if len(keys) == 3:
        k0, k1 = keys[:2]
        return [[(Ep.find_episode_cond(k0, value=v0) & Ep.find_episode_cond(k1, value=v1),
                  '%s=%.2f\n%s=%.2f' % (k0, v0, k1, v1))
                 for v0 in Ep.varied_parameters[k0]]
                for v1 in Ep.varied_parameters[k1]]

    raise ValueError('plot_traces handles 1 to 3 varied parameters, got %s' % keys)


# %%
# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_traces(Ep,
                roi=None,
                title='',
                average_by=None,
                LED_masks=None,
                traces='dFoF',
                color_led='lightskyblue',
                verbose=False):
    """
    roi: None (average all rois), a specific roi, or a set of rois (averaged)
    LED_masks: (LED_on, LED_off) boolean arrays over episodes,
               computed with find_EP_LED_on_off if None
    average_by: 'stimulus' -> one LED-off / LED-on pair over all episodes,
                otherwise one pair per stimulus condition
    """
    LED_on, LED_off = _check_LED_masks(Ep, LED_masks, verbose=verbose)
    grid = _condition_grid(Ep, average_by)
    n_rows, n_cols = len(grid), 2 * len(grid[0])

    fig, AX = pt.figure(ax_scale=(0.9, 0.95), axes=(n_cols, n_rows),
                        wspace=0.7, hspace=1.0, top=2.7)
    fig.suptitle(title)

    # baseline subtraction
    baseline = Ep.t < 0
    trace = get_trace(Ep, traces, roi)
    trace = trace - trace[:, baseline].mean(axis=1, keepdims=True)
    if traces == 'raw_fluo':
        neuropil = get_trace(Ep, 'neuropil', roi)
        neuropil = neuropil - neuropil[:, baseline].mean(axis=1, keepdims=True)

    for j, row in enumerate(grid):
        for ic, (cond, label) in enumerate(row):
            for led_i, LED in enumerate((LED_off, LED_on)):
                i = 2 * ic + led_i
                ax = AX[i] if n_rows == 1 else AX[j][i]

                if led_i == 1:  # opto window
                    ax.axvspan(-1, Ep.time_duration[0] + 1,
                               color=color_led, alpha=.3, lw=0)

                ep_cond = cond & LED
                if verbose and not np.any(ep_cond):
                    print('[plot_traces] no episode for %r, LED %s'
                          % (label, 'on' if led_i else 'off'))

                pt.plot(Ep.t, trace[ep_cond, :].mean(axis=0),
                        ax=ax, no_set=True, color='tab:green')
                if traces == 'raw_fluo':
                    pt.plot(Ep.t, neuropil[ep_cond, :].mean(axis=0),
                            ax=ax, no_set=True, color='tab:red')

                if label:
                    pt.annotate(ax, label, (0.5, 0), va='top', ha='center')

                # stimulus shading
                if Ep.protocol_name != 'protocol-grey-screen':
                    ax.axvspan(0, Ep.time_duration[0],
                               color='tab:gray', alpha=.2, lw=0)
                ax.axis('off')

    # --------------------------------------------------
    # Common formatting
    # --------------------------------------------------
    pt.set_common_ylims(AX)

    if n_rows == 1:
        last_ax, first_ax = AX[-1], AX[0]
    else:
        last_ax, first_ax = AX[0][-1], AX[0][0]

    if traces == 'raw_fluo':
        pt.annotate(last_ax, 'neuropil ', (0, 1), ha='right', color='tab:red')
        pt.annotate(last_ax, 'raw ROI fluo', (0.2, 1), ha='left', color='seagreen')
    elif traces == 'dFoF':
        pt.annotate(last_ax, 'dFoF', (0.5, 1), ha='left', color='seagreen')
    elif traces == 'Zscore_correctedFluo':
        pt.annotate(last_ax, 'Zscore(correctedFluo)', (-1, 1), ha='left', color='seagreen')

    pt.draw_bar_scales(first_ax,
                       Xbar=1,
                       Xbar_label='1s',
                       Ybar_label_format='%.2f')
    return fig


def compute_peak_values(Ep, trace='dFoF', roi=None):
    """
    Response per episode: mean over the peak window - mean over the baseline window
    (windows depend on the protocol, see below). Returns shape (n_episodes,)
    """
    signal = get_trace(Ep, trace, roi)

    # baseline blank vs baseline opto
    if Ep.protocol_name == 'protocol-grey-screen':
        baseline_t = (Ep.t > -2) & (Ep.t < -1)
        peak_t = (Ep.t > -1) & (Ep.t < Ep.time_duration[0] + 1)
    else:
        baseline_t = (Ep.t > -1) & (Ep.t < 0)
        peak_t = (Ep.t > Ep.time_duration[0] - 0.5) & (Ep.t < Ep.time_duration[0])

    return signal[:, peak_t].mean(axis=1) - signal[:, baseline_t].mean(axis=1)


def plot_variation_trace(Ep,
                        roi=None,
                        LED_masks=None,
                        color_led='lightskyblue',
                        trace='dFoF',
                        verbose=False):
    """
    Bar plot of the response (peak window - baseline window) for
    LED-off vs LED-on episodes, averaged over all stimuli.
    """
    peak_val = compute_peak_values(Ep, trace, roi)
    LED_on, LED_off = _check_LED_masks(Ep, LED_masks, verbose=verbose)
    return plot_variation_bars(peak_val[LED_off], peak_val[LED_on],
                               color_led=color_led, trace=trace, verbose=verbose)


def plot_variation_merged(Eps,
                          LED_masks_list,
                          roi=None,
                          color_led='lightskyblue',
                          trace='dFoF',
                          title='',
                          verbose=False):
    """
    Same as plot_variation_trace but pooling the episodes of several protocols
    (each protocol keeps its own baseline/peak windows).
    """
    blank, opto = [], []
    for Ep, LED_masks in zip(Eps, LED_masks_list):
        peak_val = compute_peak_values(Ep, trace, roi)
        LED_on, LED_off = _check_LED_masks(Ep, LED_masks, verbose=verbose)
        blank.append(peak_val[LED_off])
        opto.append(peak_val[LED_on])
    return plot_variation_bars(np.concatenate(blank), np.concatenate(opto),
                               color_led=color_led, trace=trace,
                               title=title, verbose=verbose)


def plot_merged_traces(Eps,
                       LED_masks_list,
                       roi=None,
                       title='',
                       traces='dFoF',
                       color_led='lightskyblue',
                       verbose=False):
    """
    LED-off / LED-on average traces pooling the episodes of several protocols.
    Episodes are aligned on stimulus onset and interpolated on a common time
    axis cropped to the time range shared by all protocols.
    Stimulus durations can differ: dark gray = shortest stimulus,
    light gray = up to the longest one.
    """
    t_start = max(Ep.t[0] for Ep in Eps)
    t_end = min(Ep.t[-1] for Ep in Eps)
    dt = min(np.median(np.diff(Ep.t)) for Ep in Eps)
    t = np.arange(t_start, t_end, dt)
    durations = [Ep.time_duration[0] for Ep in Eps]
    min_dur, max_dur = min(durations), max(durations)

    quantities = [traces, 'neuropil'] if traces == 'raw_fluo' else [traces]
    pooled = {q: {'off': [], 'on': []} for q in quantities}

    for Ep, LED_masks in zip(Eps, LED_masks_list):
        LED_on, LED_off = _check_LED_masks(Ep, LED_masks, verbose=verbose)
        for q in quantities:
            tr = get_trace(Ep, q, roi)
            tr = tr - tr[:, Ep.t < 0].mean(axis=1, keepdims=True)  # baseline subtraction
            tr = interp1d(Ep.t, tr, axis=1, bounds_error=False, fill_value=np.nan)(t)
            pooled[q]['off'].append(tr[LED_off])
            pooled[q]['on'].append(tr[LED_on])
        if verbose:
            print('[plot_merged_traces] %s: duration=%.1fs, LED off n=%i, LED on n=%i'
                  % (Ep.protocol_name, Ep.time_duration[0], LED_off.sum(), LED_on.sum()))

    fig, AX = pt.figure(ax_scale=(0.9, 0.95), axes=(2, 1),
                        wspace=0.7, hspace=1.0, top=2.7)
    fig.suptitle(title)

    for i, state in enumerate(['off', 'on']):
        ax = AX[i]
        if state == 'on':  # opto window (-1s to stim end +1s)
            ax.axvspan(-1, min_dur + 1, color=color_led, alpha=.3, lw=0)
        for q, c in zip(quantities, ['tab:green', 'tab:red']):
            data_q = np.concatenate(pooled[q][state])
            pt.plot(t, np.nanmean(data_q, axis=0), ax=ax, no_set=True, color=c)
        n = len(np.concatenate(pooled[traces][state]))
        pt.annotate(ax, 'LED %s, n=%i' % (state, n), (0.5, 0), va='top', ha='center')
        # stimulus shading
        ax.axvspan(0, min_dur, color='tab:gray', alpha=.2, lw=0)
        if max_dur > min_dur:
            ax.axvspan(min_dur, max_dur, color='tab:gray', alpha=.08, lw=0)
        ax.axis('off')

    pt.set_common_ylims(AX)
    if traces == 'raw_fluo':
        pt.annotate(AX[-1], 'neuropil ', (0, 1), ha='right', color='tab:red')
    pt.annotate(AX[-1], TRACE_LABELS[traces], (0.2, 1), ha='left', color='seagreen')
    pt.draw_bar_scales(AX[0], Xbar=1, Xbar_label='1s', Ybar_label_format='%.2f')
    return fig


def plot_variation_bars(peak_val_blank,
                        peak_val_opto,
                        color_led='lightskyblue',
                        trace='dFoF',
                        title='',
                        verbose=False):
    if trace not in VARIATION_YLABELS:
        raise ValueError('trace must be one of %s' % list(VARIATION_YLABELS))

    fig, ax = pt.figure(ax_scale=(0.9, 0.95), axes=(1, 1),
                        wspace=0.7, hspace=1.0, top=2)
    if title:
        fig.suptitle(title)

    if verbose:
        print('[plot_variation] blank: mean=%.3f (n=%i) | opto: mean=%.3f (n=%i)'
              % (np.mean(peak_val_blank), len(peak_val_blank),
                 np.mean(peak_val_opto), len(peak_val_opto)))

    pt.bar(ax=ax, x=[0, 1],
           y=[np.mean(peak_val_blank), np.mean(peak_val_opto)],
           sy=[stats.sem(peak_val_blank), stats.sem(peak_val_opto)],
           COLORS=['grey', color_led])

    pt.set_plot(ax=ax,
                xticks=[0, 1],
                xticks_labels=['Blank', 'Opto'],
                num_xticks=2,
                ylabel=VARIATION_YLABELS[trace])

    try:
        stat_res = calc_stats('Variation dFoF', peak_val_blank, peak_val_opto, debug=verbose)
        plot_stats(ax, n_groups=2, stats=stat_res, y_pos1=1, y_pos2=1)
    except Exception as e:
        print('[plot_variation] calc_stats failed: %s' % e)

    return fig, ax


# %% SESSIONS 
BASE = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments', 'opto', 'vision-survey-opto')

# opsin / opsin_cells / gcamp_cells / setup (LED color) / led model / current when known
SESSIONS = [
    # --- stgACR2 on NDNF, GCaMP on NDNF: NONE -> RECORD MORE HERE

    # --- ArchT on NDNF, GCaMP on NDNF
    dict(line='NDNF-Cre', file='2026_08_27-14-28-15.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='NDNF', setup='archt', led='M590L4', mA=900),
    dict(line='NDNF-Cre', file='2026_08_27-14-57-29.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='NDNF', setup='archt', led='M590L4', mA=900),
    dict(line='NDNF-Cre', file='2026_09_01-14-26-23.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='NDNF', setup='archt', led='M590L4', mA=900),
    dict(line='NDNF-Cre', file='2026_09_02-10-50-31.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='NDNF', setup='archt', led='M590L4', mA=900),
    dict(line='NDNF-Cre', file='2026_09_03-12-12-18.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='NDNF', setup='archt', led='M590L4', mA=900),

    # --- stgACR2 on SST, GCaMP on SST
    dict(line='SST-Cre', file='2026_11_03-16-29-10.nwb', opsin='stGtACR2', opsin_cells='SST', gcamp='SST', setup='stg',led='M450LP2', mA=900),
    dict(line='SST-Cre', file='2026_11_05-15-09-25.nwb', opsin='stGtACR2', opsin_cells='SST', gcamp='SST', setup='stg',led='M450LP2', mA=900),
    dict(line='SST-Cre', file='2026_11_05-17-05-23.nwb', opsin='stGtACR2', opsin_cells='SST', gcamp='SST', setup='stg',led='M450LP2', mA=900),

    # --- ArchT on SST, GCaMP on SST
    dict(line='SST-Cre', file='2026_11_04-15-46-20.nwb', opsin='ArchT', opsin_cells='SST', gcamp='SST', setup='archt',led='M590L4',mA=900),
    dict(line='SST-Cre', file='2026_11_05-18-17-02.nwb', opsin='ArchT', opsin_cells='SST', gcamp='SST', setup='archt',led='M590L4',mA=900),

    # --- stgACR2 on NDNF, GCaMP on PYR
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_08_26-15-34-27.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M450LP2', mA=1000),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_08_28-15-45-15.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M450LP2', mA=900),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_09_02-11-56-52.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M450LP2', mA=900),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_08_31-11-00-02.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M450LP2', mA=900),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_10_06-14-41-06.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M450LP2', mA=700),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_10_06-11-38-44.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M455L3', mA=700),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_10_05-16-58-19.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M455L3', mA=900),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_10_06-14-01-46.nwb', opsin='stGtACR2', opsin_cells='NDNF', gcamp='PYR', setup='stg', led='M455L3', mA=1000),

    # --- ArchT on NDNF, GCaMP on PYR
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_10_06-15-41-49.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='PYR', setup='archt', led='M590L4', mA=700),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_10_06-16-16-41.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='PYR', setup='archt', led='M590L4', mA=1000),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_08_31-13-30-18.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='PYR', setup='archt', led='M590L4', mA=900, note='issue with file'),
    dict(line='Thy1GCaMP-NDNF-Cre', file='2026_08_31-15-35-23.nwb', opsin='ArchT', opsin_cells='NDNF', gcamp='PYR', setup='archt', led='M590L4', mA=900, note='issue with file?'),

    # --- control animal with no opsin
    dict(line='C57BL6J', file='2026_10_02-17-18-59.nwb', opsin=None, opsin_cells=None, gcamp='PYR', setup='archt', led='M590L4', mA=900),
    dict(line='C57BL6J', file='2026_10_01-14-07-58.nwb', opsin=None, opsin_cells=None, gcamp='PYR', setup='archt', led='M590L4', mA=900),
    dict(line='C57BL6J', file='2026_10_01-15-04-50.nwb', opsin=None, opsin_cells=None, gcamp='PYR', setup='archt', led='M590L4', mA=900),
    dict(line='C57BL6J', file='2026_11_05-11-54-47.nwb', opsin=None, opsin_cells=None, gcamp='PYR', setup='archt', led='M590L4', mA=900),
    dict(line='C57BL6J', file='2026_11_05-14-09-19.nwb', opsin=None, opsin_cells=None, gcamp='PYR', setup='stg',   led='M450LP2',mA=900),
]

def session_path(s):
    return os.path.join(BASE, s['line'], 'NWBs', s['file'])


def select_sessions(**criteria):
    """e.g. select_sessions(opsin='ArchT', gcamp='PYR')"""
    return [s for s in SESSIONS if all(s.get(k) == v for k, v in criteria.items())]


def led_color(s):
    return SETUP_COLORS.get(s.get('setup'), DEFAULT_LED_COLOR)


# %%
# ------------------------------------------------------------------
# Choose session + protocol
# ------------------------------------------------------------------
file = '2026_11_05-15-09-25.nwb'
file = '2026_11_05-17-05-23.nwb'
session = select_sessions(file=file)[0]
path = session_path(session)
color = led_color(session)

# 'protocol-center-patch', 'protocol-ff_drifting_grating', 'protocol-grey-screen',
# 'protocol-natural-images', 'protocol-looming'
PROTOCOL_NAME = 'protocol-center-patch'
NEUROPIL_FACTORS = [0, 0.7, 1.0]
N_EXAMPLE_ROIS = 120

# %% load data
data = Data(path)
data.build_rawFluo()
data.build_neuropil()
data.build_correctedFluo(neuropil_correction_factor=1, verbose=VERBOSE)
data.build_Zscore(quantity='correctedFluo')
data.build_dFoF(neuropil_correction_factor=0)
data.build_visual_stim()
fix_protocol_names(data, verbose=VERBOSE)
protocol_id = get_protocol_id(data, PROTOCOL_NAME)
if VERBOSE:
    print('protocols:', list(data.protocols), '-> using', protocol_id, PROTOCOL_NAME)

# %% MEAN ALL ROIS
Ep = EpisodeData(data, protocol_id=protocol_id,
                 quantities=QUANTITIES,
                 prestim_duration=2,
                 verbose=VERBOSE)
LED_masks = find_EP_LED_on_off(Ep, verbose=VERBOSE)
title = make_title(data, Ep.protocol_name, 'mean over all valid ROIs (%s)' % data.nROIs)

plot_traces(Ep, title=title, average_by='stimulus', LED_masks=LED_masks,
            traces='raw_fluo', color_led=color, verbose=VERBOSE)

plot_traces(Ep, title=title, average_by='stimulus', LED_masks=LED_masks,
            traces='Zscore_correctedFluo', color_led=color, verbose=VERBOSE)
plot_variation_trace(Ep, LED_masks=LED_masks, color_led=color,
                    trace='Zscore_correctedFluo', verbose=VERBOSE)

for npc in NEUROPIL_FACTORS:
    data.build_dFoF(neuropil_correction_factor=npc)
    Ep = EpisodeData(data, protocol_id=protocol_id,
                     quantities=QUANTITIES,
                     prestim_duration=2,
                     verbose=VERBOSE)
    plot_traces(Ep, title=title + ', npc=%.1f' % npc, average_by='stimulus',
                LED_masks=LED_masks, traces='dFoF', color_led=color, verbose=VERBOSE)
    plot_variation_trace(Ep, LED_masks=LED_masks, color_led=color, verbose=VERBOSE)

# %% RESPONSIVE ROIS
Ep = EpisodeData(data, protocol_id=protocol_id,
                 quantities=QUANTITIES,
                 prestim_duration=2,
                 verbose=VERBOSE)
LED_masks = find_EP_LED_on_off(Ep, verbose=VERBOSE)
resp_roi, pos_roi, neg_roi = calc_responsiveness2(Ep,
                                                  data.nROIs,
                                                  alpha=0.05,
                                                  t_window=1.0,
                                                  repetition_keys=['repeat', 'contrast', 'radius'],
                                                  filtering_cond=None,
                                                  param=None,
                                                  param_type=None,
                                                  quantity="Zscore_correctedFluo")
resp_roi_indexes = np.where(resp_roi)[0]
if VERBOSE:
    print('responsive ROIs:', resp_roi_indexes)

if len(resp_roi_indexes) == 0:
    print('no responsive ROI')
else:
    title = make_title(data, Ep.protocol_name,
                       'mean over resp ROIs (%s/%s)' % (len(resp_roi_indexes), data.nROIs))
    plot_traces(Ep, roi=resp_roi_indexes, title=title, average_by='stimulus',
                LED_masks=LED_masks, traces='raw_fluo', color_led=color, verbose=VERBOSE)
    plot_traces(Ep, roi=resp_roi_indexes, title=title, average_by='stimulus',
                LED_masks=LED_masks, traces='Zscore_correctedFluo', color_led=color, verbose=VERBOSE)
    plot_variation_trace(Ep, roi=resp_roi_indexes, LED_masks=LED_masks, color_led=color,
                        trace='Zscore_correctedFluo', verbose=VERBOSE)


# %% EXAMPLE ROIS: neuropil and fluorescence time course
example_session = select_sessions(file=file)[0]
data = Data(session_path(example_session))
data.build_rawFluo()
data.build_neuropil()
data.build_dFoF(neuropil_correction_factor=0.0)
data.build_visual_stim()
fix_protocol_names(data, verbose=VERBOSE)

Ep = EpisodeData(data, protocol_id=get_protocol_id(data, PROTOCOL_NAME),
                 quantities=['dFoF', 'rawFluo', 'neuropil', 'opto'],
                 prestim_duration=2,
                 verbose=VERBOSE)
LED_masks = find_EP_LED_on_off(Ep, verbose=VERBOSE)

resp_roi, pos_roi, neg_roi = calc_responsiveness2(Ep,
                                                  data.nROIs,
                                                  alpha=0.05,
                                                  t_window=1.0,
                                                  repetition_keys=['repeat', 'contrast', 'radius'],
                                                  filtering_cond=None,
                                                  param=None,
                                                  param_type=None)
resp_roi_indexes = np.where(resp_roi)[0]
if VERBOSE:
    print('responsive ROIs:', resp_roi_indexes)

random.seed(42)
n_examples = min(N_EXAMPLE_ROIS, len(resp_roi_indexes))
for roi in random.sample(list(resp_roi_indexes), n_examples):
    plot_traces(Ep, roi=roi,
                title='%s, ROI %s,\n %s' % (data.filename, roi, Ep.protocol_name),
                average_by='stimulus', LED_masks=LED_masks,
                color_led=led_color(example_session), verbose=VERBOSE)
    plot_variation_trace(Ep, roi=roi, LED_masks=LED_masks,
                        color_led=led_color(example_session), verbose=VERBOSE)




# %% RESPONSIVE ROIS - ALL VISUAL PROTOCOLS MERGED
# opto effect during visual stimulation, regardless of the protocol
# (grey screen excluded; ROIs responsive to at least one visual protocol)
visual_protocols = [p for p in data.protocols if p != 'protocol-grey-screen']
Eps_visual, LED_masks_visual = [], []
resp_any = np.zeros(data.nROIs, dtype=bool)

for p in visual_protocols:
    Ep_p = EpisodeData(data, protocol_id=get_protocol_id(data, p),
                       quantities=QUANTITIES,
                       prestim_duration=2,
                       verbose=VERBOSE)
    Eps_visual.append(Ep_p)
    LED_masks_visual.append(find_EP_LED_on_off(Ep_p, verbose=VERBOSE))

    resp_p, _, _ = calc_responsiveness2(Ep_p,
                                        data.nROIs,
                                        alpha=0.05,
                                        t_window=1.0,
                                        repetition_keys=['repeat', 'contrast', 'radius'],
                                        filtering_cond=None,
                                        param=None,
                                        param_type=None,
                                        quantity="Zscore_correctedFluo")
    resp_any |= np.asarray(resp_p, dtype=bool)
    if VERBOSE:
        print('%s: %i responsive ROIs' % (p, np.sum(resp_p)))

resp_any_indexes = np.where(resp_any)[0]
if VERBOSE:
    print('ROIs responsive to >= 1 visual protocol:', resp_any_indexes)

if len(resp_any_indexes) == 0:
    print('no responsive ROI')
else:
    title = make_title(data, 'all visual protocols merged',
                       'mean over resp ROIs (%s/%s)' % (len(resp_any_indexes), data.nROIs))
    plot_merged_traces(Eps_visual, LED_masks_visual, roi=resp_any_indexes, title=title,
                       traces='raw_fluo', color_led=color, verbose=VERBOSE)
    plot_merged_traces(Eps_visual, LED_masks_visual, roi=resp_any_indexes, title=title,
                       traces='Zscore_correctedFluo', color_led=color, verbose=VERBOSE)
    plot_variation_merged(Eps_visual, LED_masks_visual, roi=resp_any_indexes,
                          color_led=color, trace='Zscore_correctedFluo',
                          title=title, verbose=VERBOSE)






#############################################################

# %% [markdown] ## ANALYSE ALL SESSIONS 
# %%
def session_opto_effect(s,
                        protocols='visual',
                        roi_mode='responsive',
                        trace='Zscore_correctedFluo',
                        verbose=False):
    """
    Mean response (peak - baseline, see compute_peak_values) of one session,
    for LED-off and LED-on episodes, pooled over the chosen protocols.
    protocols: 'visual' (all but grey screen) or a list of protocol names
    roi_mode : 'responsive' (responsive to >= 1 of the chosen protocols) or 'all'
    Returns a dict, or None if no responsive ROI.
    """
    data = Data(session_path(s))
    data.build_rawFluo()
    data.build_neuropil()
    data.build_correctedFluo(neuropil_correction_factor=1, verbose=verbose)
    data.build_Zscore(quantity='correctedFluo')
    data.build_dFoF(neuropil_correction_factor=0)
    data.build_visual_stim()
    fix_protocol_names(data, verbose=verbose)

    if protocols == 'visual':
        names = [p for p in data.protocols if p != 'protocol-grey-screen']
    else:
        names = [p for p in protocols if p in data.protocols]
        if verbose and len(names) < len(protocols):
            print('[%s] missing protocols:' % s['file'], set(protocols) - set(names))

    Eps, LED_masks_list = [], []
    resp = np.zeros(data.nROIs, dtype=bool)
    for p in names:
        Ep_p = EpisodeData(data, protocol_id=get_protocol_id(data, p),
                           quantities=QUANTITIES, prestim_duration=2, verbose=verbose)
        Eps.append(Ep_p)
        LED_masks_list.append(find_EP_LED_on_off(Ep_p, verbose=verbose))
        if roi_mode == 'responsive':
            resp_p, _, _ = calc_responsiveness2(Ep_p, data.nROIs,
                                                alpha=0.05, t_window=1.0,
                                                repetition_keys=['repeat', 'contrast', 'radius'],
                                                filtering_cond=None, param=None, param_type=None,
                                                quantity=trace)
            resp |= np.asarray(resp_p, dtype=bool)

    if roi_mode == 'responsive':
        rois = np.where(resp)[0]
        if len(rois) == 0:
            print('[%s] no responsive ROI -> skipped' % s['file'])
            return None
    else:
        rois = None

    off, on = [], []
    for Ep_p, (LED_on, LED_off) in zip(Eps, LED_masks_list):
        peak_val = compute_peak_values(Ep_p, trace, rois)
        off.append(peak_val[LED_off])
        on.append(peak_val[LED_on])
    off, on = np.concatenate(off), np.concatenate(on)

    return dict(session=s,
                off=np.mean(off), on=np.mean(on),
                off_sem=stats.sem(off), on_sem=stats.sem(on),
                n_off=len(off), n_on=len(on),
                n_rois=data.nROIs if rois is None else len(rois),
                n_rois_total=data.nROIs)

def session_group_label(s):
    opsin = '%s-%s' % (s['opsin'], s['opsin_cells']) if s.get('opsin') else 'no opsin'
    return '%s / GCaMP-%s (%s)' % (opsin, s.get('gcamp'), s.get('setup'))

def plot_session_scatter(results,
                         trace='Zscore_correctedFluo',
                         title='',
                         label_points=False):
    """
    x: response with LED off, y: response with LED on, one point per session
    (error bars: SEM over episodes). Points below the diagonal -> opto decreases the response.
    Paired Wilcoxon test across sessions, per group (needs >= 5 sessions to be meaningful).
    """
    results = [r for r in results if r is not None]
    groups = {}
    for r in results:
        groups.setdefault(session_group_label(r['session']), []).append(r)

    fig, ax = pt.figure(ax_scale=(1.6, 1.6))
    colors = pt.plt.get_cmap('tab10')

    for k, (label, rs) in enumerate(groups.items()):
        x = np.array([r['off'] for r in rs])
        y = np.array([r['on'] for r in rs])
        ax.errorbar(x, y,
                    xerr=[r['off_sem'] for r in rs], yerr=[r['on_sem'] for r in rs],
                    fmt='o', ms=4, lw=0.8, color=colors(k % 10),
                    label='%s, n=%i' % (label, len(rs)))
        if label_points:
            for r in rs:
                ax.annotate(r['session']['file'][:10], (r['off'], r['on']), fontsize=5)
        if len(rs) >= 2 and np.any(x != y):
            p = stats.wilcoxon(x, y).pvalue
            print('%s: n=%i sessions, Wilcoxon p=%.3g' % (label, len(rs), p))

    # identity line on equal axes
    vals = np.concatenate([[r['off'] - r['off_sem'], r['off'] + r['off_sem'],
                            r['on'] - r['on_sem'], r['on'] + r['on_sem']] for r in results])
    pad = 0.05 * (vals.max() - vals.min())
    lims = [vals.min() - pad, vals.max() + pad]
    ax.plot(lims, lims, '--', color='grey', lw=0.8, zorder=-1)
    ax.axhline(0, color='lightgrey', lw=0.5, zorder=-2)
    ax.axvline(0, color='lightgrey', lw=0.5, zorder=-2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    ylab = VARIATION_YLABELS.get(trace, trace).replace('\n', ' ')
    pt.set_plot(ax, xlabel='LED off: %s' % ylab, ylabel='LED on: %s' % ylab)
    ax.legend(fontsize=6, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    if title:
        fig.suptitle(title)
    return fig, ax

# %% compute (slow: loads every session)
#SCATTER_SESSIONS = [s for s in SESSIONS if 'issue' not in s.get('note', '')]
SCATTER_SESSIONS = select_sessions( opsin=None)

SCATTER_PROTOCOLS = 'visual'   # or e.g. ['protocol-looming']
SCATTER_ROI_MODE = 'responsive'  # or 'all'

scatter_results = []
for s in SCATTER_SESSIONS:
    try:
        r = session_opto_effect(s, protocols=SCATTER_PROTOCOLS,
                                roi_mode=SCATTER_ROI_MODE, verbose=False)
    except Exception as e:
        print('[%s] failed: %s' % (s['file'], e))
        continue
    scatter_results.append(r)
    if VERBOSE and r is not None:
        print('%s: off=%.3f, on=%.3f, ROIs %i/%i, episodes off/on %i/%i'
              % (s['file'], r['off'], r['on'], r['n_rois'], r['n_rois_total'],
                 r['n_off'], r['n_on']))

# %% plot
plot_session_scatter(scatter_results,
                     title='%s protocols, %s ROIs' % (SCATTER_PROTOCOLS, SCATTER_ROI_MODE))
