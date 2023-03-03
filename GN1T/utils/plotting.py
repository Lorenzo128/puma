"""
    Plotting functions for validation of inputs and evaluating performance.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.interpolate import pchip

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import Divider, Size

import seaborn as sns

import atlas_mpl_style as ampl
from atlasify import AtlasStyle

import utils.evaluation as ev

# ---------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------
def set_style(path='../utils/custom.mplstyle'):
    ampl.use_atlas_style()
    plt.style.use(path)


atlas_style = AtlasStyle(
    atlas='Simulation Internal',
    offset=7,
    indent=7,
    font_size=13,
    label_font_size=13,
    sub_font_size=10,
    enlarge=1
)
# ---------------------------------------------------------------------
# plotting utils
# ---------------------------------------------------------------------
def get_plot(figsize=(5, 4.5), ratio_panel=False):
    fig = Figure(figsize=figsize, constrained_layout=True)
    if not ratio_panel:
        ax = fig.subplots()
        return fig, ax
    else:
        ax, ratio = fig.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [2.5, 1]}, 
            sharex=True
        )
        return fig, (ax, ratio)


def hist_leg(ax, loc='upper right', fontsize=None, ncol=1, handles=None, labels=None):

    if not handles and not labels:
        handles, labels = ax.get_legend_handles_labels()

        # remove errors from errorbars
        handles = [h[0] if isinstance(h, matplotlib.container.ErrorbarContainer) else h for h in handles]

        # turn hist box into line
        handles = [matplotlib.lines.Line2D([], [], c=h.get_edgecolor(), ls=h.get_linestyle()) 
                   if isinstance(h, matplotlib.patches.Polygon) else h for h in handles]

    
    ax.legend(handles=handles, labels=labels, loc=loc, fontsize=fontsize, ncol=ncol)


def get_fc_text(model_configs, flavour, newline=False):
    x = ev.flav_rej(flavour, return_same=False)
    fxs = set([m[f'f{x}'] for m in model_configs.values()])
    s = ''
    if len(fxs) == 1:
        s += '\n' if newline else ', '
        s += f"$f_{x} = {list(fxs)[0]}$"
    return s
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# Jet classification performance
# ---------------------------------------------------------------------
def plot_tag_discriminant(df_A, df_B, target_flavour=5, 
                          log=True, density=True,
                          bins=100, x_range=(-10, 15), 
                          yf=1.0, yi=1,
                          effs=[0.5, 0.7],
                          desc='', labels=('', ''),
                          save_path=None):
    po = {
        'bins': bins, 
        'range': x_range,
        'histtype': 'step',
        'lw':2,
        'density': density,
        'ls': (0, (1, 1)),
        'log': log
    }

    disc = ev.get_disc_variable(target_flavour)

    fig, ax = get_plot(figsize=(8, 4))

    ax.hist(df_A.loc[df_A.flavour == 5, disc], label=labels[0]+' $b$-jets', color='seagreen', **po)
    ax.hist(df_A.loc[df_A.flavour == 4, disc], label=labels[0]+' $c$-jets', color='gold', **po)
    ax.hist(df_A.loc[df_A.flavour == 0, disc], label=labels[0]+' $l$-jets', color='cornflowerblue', **po)

    po['ls'] = 'solid'
    ax.hist(df_B.loc[df_B.flavour == 5, disc], label=labels[1]+' $b$-jets', color='seagreen', **po)
    ax.hist(df_B.loc[df_B.flavour == 4, disc], label=labels[1]+' $c$-jets', color='gold', **po)
    ax.hist(df_B.loc[df_B.flavour == 0, disc], label=labels[1]+' $l$-jets', color='cornflowerblue', **po)
    
    # model A WPs
    y_min, y_max = ax.get_ylim()
    y_max *= yf

    for i, eff in enumerate(effs):
        cut = ev.get_wp_cut_from_df(df_A, eff, target_flavour)
        y_bar = (1.25 + 1.0*i) * y_max
        if log:
            y_bar *= 2**i
        ax.plot([cut, cut], [0, y_bar], ls='dashed', color='black', lw=1)

    for i, eff in enumerate(effs):
        cut = ev.get_wp_cut_from_df(df_B, eff, target_flavour)
        y_bar = (1.25 + 1.0*i) * y_max
        if log:
            y_bar *= 2**i
        ax.plot([cut, cut], [0, y_bar], ls='solid', color='black', lw=1)
        if log:
            y_bar = y_bar - 0.5 * y_bar
        else:
            y_bar = y_bar - 0.015

        ax.annotate(fr'{round(eff*100)}% WP', xy=(cut+0.25, y_bar), fontsize=9)#, bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_xlabel(f'${disc}$'); ax.set_ylabel('a.u.')
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_min, y_max*yi)
    
    if log:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*10)
    
    hist_leg(ax, ncol=2, fontsize=11)
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path)
    else:
        return fig

def plot_tag_discriminant_corr(df_A, df_B, target_flavour=5, 
                               log=True, density=True,
                               bins=100, x_range=(-10, 15), 
                               yf=1.0, yi=1,
                               effs=[0.5, 0.7],
                               desc='', labels=('', ''),
                               save_path=None):
    po = {
        'bins': bins, 
        'range': x_range,
        'histtype': 'step',
        'lw':2,
        'density': density,
        'ls': (0, (1, 1)),
        'log': log
    }

    disc = ev.get_disc_variable(target_flavour)

    # only look at jets with at least 1 track
    df_A = ev.remove_jets_1trk(df_A)
    df_B = ev.remove_jets_1trk(df_B)

    df_A_bjets = df_A.loc[df_A.flavour == 5, disc]
    df_A_cjets = df_A.loc[df_A.flavour == 4, disc]
    df_A_ljets = df_A.loc[df_A.flavour == 0, disc]

    df_B_bjets = df_B.loc[df_B.flavour == 5, disc]
    df_B_cjets = df_B.loc[df_B.flavour == 4, disc]
    df_B_ljets = df_B.loc[df_B.flavour == 0, disc]

    # regplot
    fig, ax = get_plot(figsize=(8, 4))

    sns.regplot(x=df_A_bjets, y=df_B_bjets, ax=ax, label='$b$-jets', color='seagreen')
    sns.regplot(x=df_A_cjets, y=df_B_cjets, ax=ax, label='$c$-jets', color='gold')
    sns.regplot(x=df_A_ljets, y=df_B_ljets, ax=ax, label='$l$-jets', color='cornflowerblue')
    ax.add_artist(lines.Line2D(x_range, x_range, color = 'red', alpha = 0.4, ls = '--', lw = 1))

    ax.set_xlabel(f'{labels[0]} ${disc}$')
    ax.set_ylabel(f'{labels[1]} ${disc}$')
    ax.set_xlim(x_range)
    ax.set_ylim(x_range)

    hist_leg(ax, ncol=1, fontsize=11)
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path)

    # histplot
    fig, ax = get_plot(figsize=(8, 4))

    sns.histplot(x=df_A_bjets, y=df_B_bjets, ax=ax, label='$b$-jets', cbar=True, cbar_kws = {'label':'$b$-jets'}, color='seagreen')
    sns.histplot(x=df_A_cjets, y=df_B_cjets, ax=ax, label='$c$-jets', cbar=True, cbar_kws = {'label':'$c$-jets'}, color='gold')
    sns.histplot(x=df_A_ljets, y=df_B_ljets, ax=ax, label='$l$-jets', cbar=True, cbar_kws = {'label':'$l$-jets'}, color='cornflowerblue')
    ax.add_artist(lines.Line2D(x_range, x_range, color = 'red', alpha = 0.4, ls = '--', lw = 1))

    ax.set_xlabel(f'{labels[0]} ${disc}$')
    ax.set_ylabel(f'{labels[1]} ${disc}$')
    ax.set_xlim(x_range)
    ax.set_ylim(x_range)

    # not quite sure why legend is not showing here :(
    hist_leg(ax, ncol=1, fontsize=20)
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path.replace('.png', '_hist.png'))
    else:
        return fig

def plot_model_comparison(df_A, df_B, wp_cut_A, wp_cut_B,
                          var='pt',
                          target_flavour=5,
                          log=True, density=False,
                          bins=100, x_range=(-10, 15),
                          ratio_y_range=(0, 2),
                          yf=1.0, yi=10,
                          effs=[0.5, 0.7],
                          desc='', labels=('', ''),
                          save_path=None):
    po = {
        'bins' : bins, 
        'range' : x_range,
        'histtype' : 'step',
        'lw' : 2,
        'density' : density,
        'ls' : (0, (1, 1)),
        'log' : log
    }
    flavours = ['b', 'c', 'l']
    colors = ['seagreen', 'gold', 'cornflowerblue']
    lss = {
        'A' : 'solid',
        'B' : 'dashed',
    }

    disc = ev.get_disc_variable(target_flavour)

    # only look at jets with more than 1 track
    df_A = ev.remove_jets_1trk(df_A)
    df_B = ev.remove_jets_1trk(df_B)

    if var == 'pt':
        # denominators
        njets_b = len(df_A.loc[df_A.flavour == 5])
        njets_c = len(df_A.loc[df_A.flavour == 4])
        njets_l = len(df_A.loc[df_A.flavour == 0])
        njets_tau = len(df_A.loc[df_A.flavour == 15])
        print(f'\tInitial jets ({save_path.split("/")[-1].split("_")[0]}):')
        print(f'\t\tb-jets: {njets_b} ')
        print(f'\t\tc-jets: {njets_c} ')
        print(f'\t\tl-jets: {njets_l} ')
        print(f'\t\ttau-jets: {njets_tau} ')

    # jets passing A and failing B (and vice-versa)
    df_A_not_B = ev.select_pass_A_fail_B(df_A, df_B, wp_cut_A, wp_cut_B, disc=disc)
    df_B_not_A = ev.select_pass_A_fail_B(df_B, df_A, wp_cut_B, wp_cut_A, disc=disc)
    # plot jets passing A/B for comparison
    df_A = df_A.loc[df_A[disc] > wp_cut_A]
    df_B = df_B.loc[df_B[disc] > wp_cut_B]
    if var == 'pt':
        njets_pass_A_b = len(df_A.loc[df_A.flavour == 5])
        njets_pass_A_c = len(df_A.loc[df_A.flavour == 4])
        njets_pass_A_l = len(df_A.loc[df_A.flavour == 0])
        njets_pass_A_tau = len(df_A.loc[df_A.flavour == 15])
        njets_pass_B_b = len(df_B.loc[df_B.flavour == 5])
        njets_pass_B_c = len(df_B.loc[df_B.flavour == 4])
        njets_pass_B_l = len(df_B.loc[df_B.flavour == 0])
        njets_pass_B_tau = len(df_B.loc[df_B.flavour == 15])
        njets_pass_A_not_B_b = len(df_A_not_B.loc[df_A_not_B.flavour == 5])
        njets_pass_A_not_B_c = len(df_A_not_B.loc[df_A_not_B.flavour == 4])
        njets_pass_A_not_B_l = len(df_A_not_B.loc[df_A_not_B.flavour == 0])
        njets_pass_A_not_B_tau = len(df_A_not_B.loc[df_A_not_B.flavour == 15])
        njets_pass_B_not_A_b = len(df_B_not_A.loc[df_B_not_A.flavour == 5])
        njets_pass_B_not_A_c = len(df_B_not_A.loc[df_B_not_A.flavour == 4])
        njets_pass_B_not_A_l = len(df_B_not_A.loc[df_B_not_A.flavour == 0])
        njets_pass_B_not_A_tau = len(df_B_not_A.loc[df_B_not_A.flavour == 15])
        print(f'\tjets passing {labels[0]}:')
        print(f'\t\tb-jets: {njets_pass_A_b} ({100*njets_pass_A_b/njets_b:.2f}%)')
        print(f'\t\tc-jets: {njets_pass_A_c} ({100*njets_pass_A_c/njets_c:.2f}%)')
        print(f'\t\tl-jets: {njets_pass_A_l} ({100*njets_pass_A_l/njets_l:.2f}%)')
        print(f'\t\ttau-jets: {njets_pass_A_tau} ({100*njets_pass_A_tau/njets_tau:.2f}%)')
        print(f'\tjets passing {labels[1]}:')
        print(f'\t\tb-jets: {njets_pass_B_b} ({100*njets_pass_B_b/njets_b:.2f}%)')
        print(f'\t\tc-jets: {njets_pass_B_c} ({100*njets_pass_B_c/njets_c:.2f}%)')
        print(f'\t\tl-jets: {njets_pass_B_l} ({100*njets_pass_B_l/njets_l:.2f}%)')
        print(f'\t\ttau-jets: {njets_pass_B_tau} ({100*njets_pass_B_tau/njets_tau:.2f}%)')
        print(f'\tjets passing {labels[0]} and failing {labels[1]}:')
        print(f'\t\tb-jets: {njets_pass_A_not_B_b} ({100*njets_pass_A_not_B_b/njets_b:.2f}%)')
        print(f'\t\tc-jets: {njets_pass_A_not_B_c} ({100*njets_pass_A_not_B_c/njets_c:.2f}%)')
        print(f'\t\tl-jets: {njets_pass_A_not_B_l} ({100*njets_pass_A_not_B_l/njets_l:.2f}%)')
        print(f'\t\ttau-jets: {njets_pass_A_not_B_tau} ({100*njets_pass_A_not_B_tau/njets_tau:.2f}%)')
        print(f'\tjets passing {labels[1]} and failing {labels[0]}:')
        print(f'\t\tb-jets: {njets_pass_B_not_A_b} ({100*njets_pass_B_not_A_b/njets_b:.2f}%)')
        print(f'\t\tc-jets: {njets_pass_B_not_A_c} ({100*njets_pass_B_not_A_c/njets_c:.2f}%)')
        print(f'\t\tl-jets: {njets_pass_B_not_A_l} ({100*njets_pass_B_not_A_l/njets_l:.2f}%)')
        print(f'\t\ttau-jets: {njets_pass_B_not_A_tau} ({100*njets_pass_B_not_A_tau/njets_tau:.2f}%)')

    fig, (ax, ratio_ax) = get_plot(figsize=(8, 4), ratio_panel=True)

    # main plot
    print(f'\tplotting: {var}')
    ns, bins, patches = {}, {}, {}
    for f, c, in zip(flavours, colors):
        k = f'A_{f}'
        po['ls'] = lss['A']
        ns[k], bins[k], patches[k] = ax.hist(df_A.loc[df_A.flavour == ev.flav(f), var], label=labels[0]+f' ${f}$-jets', color=c, **po)
        k = f'B_{f}'
        po['ls'] = lss['B']
        ns[k], bins[k], patches[k] = ax.hist(df_B.loc[df_B.flavour == ev.flav(f), var], label=labels[1]+f' ${f}$-jets', color=c, **po)

    # ratios
    ratios = {}
    ratio_ax.axhline(y = 1, color = 'red', alpha = 0.4, ls = '-', lw = 1)
    for f, c in zip(flavours, colors):
        ratios[f'A_{f}'] = ns[f'A_{f}'] / ns[f'B_{f}']
        # maybe check for div-by-zero!
        # ratios[k][np.isnan(ratios[k])] = 1
        ratio_ax.stairs(
            ratios[f'A_{f}'],
            bins['A_b'],
            color = c,
            lw = 2,
            ls = lss['A']
        )

    # formatting
    y_min, y_max = ax.get_ylim()
    y_max *= yf

    ratio_ax.set_xlabel(f'{var}')
    ratio_ax.set_ylabel(f'Ratio to {labels[1]}', loc='center')
    ratio_ax.set_ylim(ratio_y_range)
    ax.set_ylabel('Number of jets')
    ax.set_xlim(x_range)
    ax.set_ylim(y_min, y_max*yi)
    fig.align_ylabels()
    
    if log:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*10)
    
    hist_leg(ax, ncol=3, fontsize=11)
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path)

    # Plot A not B
    fig, ax = get_plot(figsize=(8, 4))

    for f, c, in zip(flavours, colors):
        k = f'A_not_B_{f}'
        po['ls'] = 'solid'
        ns[k], bins[k], patches[k] = ax.hist(df_A_not_B.loc[df_A_not_B.flavour == ev.flav(f), var], label=f'{labels[0]} not {labels[1]} ${f}$-jets', color=c, **po)

    # formatting
    y_min, y_max = ax.get_ylim()
    y_max *= yf
    ax.set_xlabel(f'{var}')
    ax.set_ylabel('Number of jets')
    ax.set_xlim(x_range)
    ax.set_ylim(y_min, y_max*yi)
    fig.align_ylabels()
    
    if log:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*10)
    
    hist_leg(ax, ncol=1, fontsize=11)
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path.replace('.pdf', '_AnotB.pdf'))

    # Plot B not A
    fig, ax = get_plot(figsize=(8, 4))

    for f, c, in zip(flavours, colors):
        k = f'B_not_A_{f}'
        po['ls'] = 'solid'
        ns[k], bins[k], patches[k] = ax.hist(df_B_not_A.loc[df_B_not_A.flavour == ev.flav(f), var], label=f'{labels[1]} not {labels[0]} ${f}$-jets', color=c, **po)

    # formatting
    y_min, y_max = ax.get_ylim()
    y_max *= yf
    ax.set_xlabel(f'{var}')
    ax.set_ylabel('Number of jets')
    ax.set_xlim(x_range)
    ax.set_ylim(y_min, y_max*yi)
    fig.align_ylabels()
    
    if log:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*10)
    
    hist_leg(ax, ncol=1, fontsize=11)
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path.replace('.pdf', '_BnotA.pdf'))
    else:
        return fig


def plot_rejection_rocs(models, config, denominator,
                        target_flavour=5, 
                        x_range=(0.5, 1.0),
                        y_range_left=None,
                        y_range_right=None,
                        ratio_y_range_left=None,
                        ratio_y_range_right=None,
                        desc='', save_path=None):
    
    efficiencies=np.linspace(*x_range, 100)

    x = 'b' if target_flavour == 5 else 'c'
    y = 'c' if target_flavour == 5 else 'b'

    config['x_range'] = (efficiencies[0], efficiencies[-1])
    
    # process models
    eff_dfs = {
        model: ev.get_roc_rejections(
            df,
            target_flavour=target_flavour, 
            efficiencies=efficiencies
        ) 
        for model, df in models.items()
    }

    # make axes
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 4.5), 
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0)

    # make rejection plots
    plot_roc(
        eff_dfs, x, y, config, 
        axes=axes[:,0], fig=fig, target_flavour=target_flavour, 
        desc=desc, denominator=denominator,
        y_range=y_range_left,
        ratio_y_range=ratio_y_range_left
    )
    plot_roc(
        eff_dfs, x, 'l', config, 
        axes=axes[:,1], fig=fig, target_flavour=target_flavour, 
        desc=desc, denominator=denominator,
        y_range=y_range_right,
        ratio_y_range=ratio_y_range_right
    )

    if save_path:
        fig.savefig(save_path)
    
    
def plot_roc(eff_dfs, x, y, config, denominator, 
             axes=None, fig=None, target_flavour=5, desc='', y_range=None, ratio_y_range=None):
    
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), 
                                 gridspec_kw={'height_ratios': [2.5, 1]}, 
                                 sharex=True)

    x_var = f'{x}_eff'
    y_var = f'{y}_rej'
    y_var_err = f'{y}_err'
    den_df = eff_dfs[denominator]
    den_rel_err = den_df[y_var_err] / den_df[y_var]

    # plot each curve
    for model, df in eff_dfs.items():

        style = config['models'][model]['style']

        # main
        xs = df[x_var]; ys = df[y_var]; yerr = df[y_var_err]
        axes[0].semilogy(xs,  ys, **style)
        down = ys - yerr; up = ys + yerr
        axes[0].fill_between(
            xs, down, up, color=style['color'], alpha=0.3, zorder=2
        )

        # ratio plot
        denom_xs = eff_dfs[denominator][x_var]
        denom_ys = eff_dfs[denominator][y_var]
        axes[1].plot(xs, ys / denom_ys, **style)
        f_num = pchip(xs, ys); f_denom = pchip(denom_xs, denom_ys)
        ratio = f_num(xs) / f_denom(xs)
        down, up = ev.get_ratio_err(ys, yerr, ratio)
        axes[1].fill_between(xs, down, up, alpha=0.3, zorder=1, color=style['color'])

    # style
    axes[0].set_xlim(config['x_range'])
    if y_range:
        axes[0].set_ylim(map(float, y_range))
    if ratio_y_range:
        axes[1].set_ylim(ratio_y_range)
    if y == 'l':
        y = 'Light'
    else:
        y = f"${y}$"
    axes[1].set_xlabel(f'${x}$-jet tagging efficiency')
    axes[0].set_ylabel(f'{y}-jet rejection', labelpad=4)
    axes[1].set_ylabel(f'Ratio to {denominator}')
    fig.align_ylabels(axes)
    axes[0].legend()
    desc += get_fc_text(config['models'], target_flavour)
    atlas_style.apply(axes=axes[0], subtext=desc)

    return axes


def plot_step_hist(ax, xs, x_bins, ys, yerr, bin_width, style):

    # plot main lines
    ax.errorbar(xs, ys, xerr=bin_width/2, **style, fmt="none")
    
    # handle and legend
    handle = matplotlib.lines.Line2D([], [], **style)
    label = style['label']

    # up and down variations
    down = ys - yerr; down = np.concatenate((down[:1], down))
    up = ys + yerr; up = np.concatenate((up[:1], up))

    # plot errors
    ax.fill_between(
        x_bins, down, up, color=style['color'], alpha=0.3, zorder=2, step="pre", edgecolor="none",
    )

    return handle, label


def plot_differential_eff_or_rej(
    models, config,
    target_flavour,
    eff_flavour,
    x_range, x_bins, x_var='pt', 
    desc='',
    eff=None,
    flat_per_bin=False,
    y_range=(0, 1.2),
    ratio_y_range=(0, 3),
    denominator=None,
    wp=None,
    wp_cut=None,
    save_path=None,
    ):
    if wp and (eff or flat_per_bin):
        raise ValueError("Can't use wp with eff or flat_per_bin=True")
    if wp and wp_cut:
        raise ValueError("Either provide a wp in prcent or directly provide the cut")

    # create bins
    bin_width = (x_range[1] - x_range[0]) / x_bins
    x_bins = np.linspace(x_range[0], x_range[1], x_bins+1)
    xs = (x_bins[1:] + x_bins[:-1]) / 2 # get bin midpoints

    # compute efficiencies
    res = ev.get_effs_by_var(
        config, models, x_bins, x_var, target_flavour, eff_flavour, eff=eff, wp=wp, flat_per_bin=flat_per_bin
    )
    
    # ------------------------------------
    # go in wrapper
    fig, (ax, ratio_ax) = get_plot(ratio_panel=True)
    # ------------------------------------
    
    plt_handles = []; plt_labels = []

    # main plot
    for model, df in res.items():
        style = config['models'][model]['style']
        handle, label = plot_step_hist(ax, xs, x_bins, df['eff'],  df['err'], bin_width, style)
        plt_handles.append(handle); plt_labels.append(label)
    
    # ratio plots
    den_df = res[denominator]
    den_rel_err = den_df['err'] / den_df['eff']
    for model, df in res.items():
        style = config['models'][model]['style']
        ys = df['eff']; yerr = df['err']
        ratio = ys/den_df['eff']
        ratio_ax.errorbar(xs, ratio, xerr=bin_width/2, **style, fmt="none")
        down, up = ev.get_ratio_err(ys, yerr, ratio, binned=True)
        ratio_ax.fill_between(
            x_bins, down, up, color=style['color'], alpha=0.3, zorder=2, step="pre", edgecolor="none",
        )

    # formatting
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    #ratio_ax.set_ylim(ratio_y_range)
    if x_var == 'pt':
        ratio_ax.set_xlabel('Jet $p_T$ [GeV]')
    elif x_var == 'eta':
        ratio_ax.set_xlabel('Jet $\eta$')
    else:
        ratio_ax.set_xlabel(x_var)
    
    hist_leg(ax, handles=plt_handles, labels=plt_labels)

    desc += get_fc_text(config['models'], target_flavour)
    if eff:
        y_label = f'${ev.flav(target_flavour)}$-jet tagging efficiency'
        flav = ev.flav(eff_flavour, light=True)
        if len(flav) == 1:
            flav = f"${flav}$"
        eff_str = f'{flav}-jet rejection of {1/eff:.0f}'
    elif wp:
        y_label = f'${ev.flav(eff_flavour)}$-jet tagging efficiency'
        eff_str = f'{wp}% $t\overline{{t}}$ $b$-eff'
        desc += f'\nFixed {eff_str}'
    elif wp_cut:
        y_label = f'${ev.flav(eff_flavour)}$-jet tagging efficiency'
        desc += f'\nFixed cut {wp_cut}'
    
    if flat_per_bin:
        desc += f'\n{eff_str} per bin'
    elif not (wp or wp_cut):
        desc += f'\nInclusive {eff_str}'

    ax.set_ylabel(y_label)
    ratio_ax.set_ylabel(f'Ratio to {denominator}', loc='center')
    fig.align_ylabels()
    atlas_style.apply(axes=ax, subtext=desc)

    if save_path:
        fig.savefig(save_path)


def plot_fc_scan(
    models,
    config,
    target_flavour=5,
    eff=0.7,
    x_range=[0, 1],
    y_range=[0, 1],
    desc="",
    save_path=None,
):

    bc_flav = ev.flav_rej(target_flavour, return_same=False)

    # compute the scans through fc for each model
    fc_space = ev.get_fc_space()
    fc_scans = ev.get_fc_scans(config, models, target_flavour=target_flavour, eff=eff, fc_space=fc_space)

    # ------------------------------
    # set up the plot
    fig = Figure(figsize=(5, 4.5))

    # The first & third items are for padding and the second items are for the
    # axes. Sizes are in inches.
    h = [Size.Fixed(0.8), Size.Scaled(1.), Size.Fixed(.2)]
    v = [Size.Fixed(0.5), Size.Scaled(1.), Size.Fixed(.3)]

    # The width and height of the rectangle are ignored.
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    # ------------------------------
    
    # plot each model
    for model, df in fc_scans.items():

        # get points
        xys = df.values
        xs, ys = xys[:, 0], xys[:, 1]

        # first plot the fc markers so that we can modify the curves later
        fx = config["models"][model][f"f{bc_flav}"]
        if fx == 'auto' or fx == 'auto_pt':
            fx_idx = ev.get_optimal_fc(df, fc_space)[0]
        else:
            fx_idx = np.argmin(np.abs(fc_space - fx))

        ax.plot(
            xs[fx_idx],
            ys[fx_idx],
            marker="x",
            color=config["models"][model]["style"]["color"],
            markersize=7,
            markeredgewidth=1.5,
        )

        # plot the fc scan, don't exceed y range
        keep_idx = ys < y_range[0] + (y_range[1] - y_range[0])*0.75
        xs = xs[keep_idx]; ys = ys[keep_idx]
        ax.plot(xs, ys, **config["models"][model]["style"])

    # formatting
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(f"${bc_flav}$-efficiency")
    ax.set_ylabel("$l$-efficiency")
    ax.legend(loc="upper right")

    # atlas label
    desc += f"\n{eff:.0%} $" + ev.flav(target_flavour) + "$-jet WP"
    atlas_style.apply(axes=ax, subtext=desc)

    # save plot
    if save_path:
        fig.savefig(save_path, bbox_inches='')
# ---------------------------------------------------------------------
