"""
Script to produce performance plots for the GNN

"""

import os
import yaml
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
import scipy
from scipy.interpolate import pchip

import utils.evaluation as ev
import utils.plotting as pu
import utils.track_origins as to
import utils.vertexing as vert


def parse_args():
    """
    Argument parser for the plotting script.
    """
    parser = argparse.ArgumentParser(description="Make performance plots.")

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help='Path to plotting config file.'
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=False,
        type=str,
        help='Name of output dir where the plots will be saved.'
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        type=str, 
        help='Parent dir of output dir, useful for collecting the results for several plotting jobs.'
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        required=False,
        action='store_true',
        help='Overwrite previous named output dir.'
    )
    parser.add_argument(
        "--sample",
        required=False,
        type=str, 
        choices=['ttbar', 'zprime'], 
        help='Make plots for only ttbar or zprime, instead of both (default).'
    )
    parser.add_argument(
        "-n",
        "--num_jets",
        required=False,
        type=float,
        help='Use this many jets to fill plots.'
    )

    args = parser.parse_args()
    return args


def plot_discriminants(models, config):
    """
    Produce tagger discrimant plots
    """

    print(f" - Plot discriminants")

    sample     = config['sample']
    out_dir    = config['out_dir']
    sample_str = config['sample_str']
    mc         = config['models']

    for plot_config in config['discriminant_plots']:
        model_A, model_B = plot_config['models']
        label_A = mc[model_A]['style']['label']; label_B = mc[model_B]['style']['label']
        mcs = {m: mc[m] for m in plot_config['models']}
        
        kwargs = plot_config['args']
        kwargs['labels'] = [label_A, label_B]
        kwargs['desc'] = sample_str + pu.get_fc_text(mcs, kwargs['target_flavour'])
        fname = f'{sample}_score_{model_A}_{model_B}_'
        fname += ev.flav(kwargs['target_flavour']) + 'tag'
        
        pu.plot_tag_discriminant(
            models[model_A], models[model_B],
            **kwargs,
            save_path=os.path.join(out_dir, f"{fname}.pdf")
        )


def plot_rocs(models, config):
    """
    Produce ROC curves
    """

    print(f" - Plot rocs")
    
    sample     = config['sample']
    out_dir    = config['out_dir']
    sample_str = config['sample_str']

    for plot_config in config['roc_plots']:
        
        if config['name'] != plot_config['sample']:
            continue
        
        kwargs = plot_config['args']
        fname = f'{sample}_roc_' + ev.flav(kwargs['target_flavour']) + 'tag'
        pu.plot_rejection_rocs(
            models, config, 
            **kwargs, 
            desc=sample_str,
            save_path=os.path.join(out_dir, f'{fname}.pdf')
        )


def plot_fc_scans(models, config):
    """
    Plot efficiencies for a scan through different values of fc, 
    with a fixed efficiency.
    """

    print(f" - Plot fc scans")

    sample     = config['sample']
    out_dir    = config['out_dir']

    for plot_config in config['fc_scans']:
        
        if config['name'] != plot_config['sample']:
            continue
        
        kwargs = plot_config['args']
        kwargs['desc'] = config['sample_str']
        fname = sample + '_fcscan_' + ev.flav(kwargs['target_flavour']) + 'tag'
        kwargs['save_path'] = os.path.join(out_dir, f'{fname}.pdf')

        pu.plot_fc_scan(models, config, **kwargs)


def plot_differential_performance(models, config):
    """
    Plot the eff/rej as a function of a variable
    """

    print(f" - Plot differential performance")

    sample     = config['sample']
    out_dir    = config['out_dir']

    for plot_config in config['differential_performance']:
        
        if config['name'] != plot_config['sample']:
            continue

        kwargs = plot_config['args']
        kwargs['desc'] = config['sample_str']

        target_str = ev.flav(kwargs['target_flavour'])
        eff_str = ev.flav(kwargs['eff_flavour'])

        if kwargs.get('wp'):
            fname = f'{sample}_fixed_{eff_str}eff_'
        else:
            fname = f'{sample}_incl_{eff_str}eff_'
        fname += 'by_' + kwargs['x_var']
        fname += f'_{target_str}tag'
        
        flat_per_bin = kwargs.pop('flat_per_bin')
        kwargs['save_path'] = os.path.join(out_dir, f'{fname}.pdf')
        pu.plot_differential_eff_or_rej(models, config, flat_per_bin=False, **kwargs)

        if flat_per_bin:
            kwargs['save_path'] = kwargs['save_path'].replace('incl', 'flat')
            pu.plot_differential_eff_or_rej(models, config, flat_per_bin=True, **kwargs)


def plot_model_comparisons(models, config):
    """
    Produce distributions for jets failing model_A and passing model_B
    """

    print(f" - Plot model comparisons")

    # only compare two models
    if len(models) != 2:
        raise ValueError(f'Model comparisons only supported for 2 models at a time; you provided {len(models)}')

    sample     = config['sample']
    out_dir    = config['out_dir']
    sample_str = config['sample_str']
    mc         = config['models']
    wp         = config['comparison_wp']

    model_A, model_B = config['models']
    label_A = mc[model_A]['style']['label']; label_B = mc[model_B]['style']['label']
    mcs = {m: mc[m] for m in config['models']}

    # ensure we are looking at the same jets
    if not models[model_A]['pt'].equals(models[model_B]['pt']):
        raise ValueError(f'Models {model_A} and {model_B} are not looking at the same jets!')

    for plot_config in config['model_comparisons']:
        if sample != plot_config['sample']:
            continue
        kwargs = plot_config['args']
        kwargs['labels'] = [label_A, label_B]
        kwargs['desc'] = sample_str + pu.get_fc_text(mcs, kwargs['target_flavour'])
        if plot_config['var'] == 'pt':
            # compare discriminants once
            fname = f'{sample}_score_{model_A}_{model_B}_'
            fname += ev.flav(kwargs['target_flavour']) + 'tag'
            pu.plot_tag_discriminant(
                models[model_A], models[model_B],
                target_flavour = kwargs['target_flavour'],
                x_range=[-15, 25], effs=kwargs['effs'],
                desc=kwargs['desc'], labels=kwargs['labels'],
                save_path=os.path.join(out_dir, f"{fname}.pdf")
            )
            pu.plot_tag_discriminant_corr(
                models[model_A], models[model_B],
                target_flavour = kwargs['target_flavour'],
                x_range=[-15, 25], effs=kwargs['effs'],
                desc=kwargs['desc'], labels=kwargs['labels'],
                save_path=os.path.join(out_dir, f"{fname}_corr.png")
            )
        # compare relevant variable
        fname = f'{sample}_{model_A}_{model_B}_{plot_config["var"]}_'
        fname += ev.flav(kwargs['target_flavour']) + 'tag'
        wp_cut_A = config['models'][model_A]['wps'][wp]
        wp_cut_B = config['models'][model_B]['wps'][wp]
        pu.plot_model_comparison(
            models[model_A], models[model_B],
            wp_cut_A, wp_cut_B,
            var = plot_config['var'],
            **kwargs,
            save_path=os.path.join(out_dir, f"{fname}.pdf")
        )


def make_jet_plots(config):
    """
    Make jet classification performance plots
    """

    # load info info
    config, models = ev.load_models(config)

    # plot discriminat comparison
    if 'discriminant_plots' in config:
        plot_discriminants(models, config)

    # plot ROC 
    if 'roc_plots' in config:
        plot_rocs(models, config)

    # plot fc scans
    if 'fc_scans' in config:
        plot_fc_scans(models, config)

    # pt plots
    if 'differential_performance' in config:
        plot_differential_performance(models, config)

    # model comparison plots
    if 'model_comparisons' in config:
        plot_model_comparisons(models, config)


def main():
    pu.set_style("/sdf/home/l/losanti/puma/GN1T/utils/custom.mplstyle")
    
    # parse args
    args = parse_args() 

    # load config
    config = ev.load_config(args.config)

    # out dir
    base_out_dir = config['plot_out_dir']
    if args.version:
        base_out_dir = os.path.join(base_out_dir, args.version)
    if args.dir:
        base_out_dir = os.path.join(base_out_dir, args.dir)
    else:
        config_basename = os.path.basename(args.config)
        outname = os.path.splitext(config_basename)[0]
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        base_out_dir = os.path.join(base_out_dir, f"{outname}__{timestamp}")

    print('Plotting in:', base_out_dir)
    os.makedirs(base_out_dir, exist_ok=args.overwrite)

    # add args to config
    config['num_jets']  = int(args.num_jets) if args.num_jets is not None else -1
    config['base_out_dir'] = base_out_dir

    # make plots for a single sample
    if args.sample is not None:
        config['samples'] = {args.sample: config['samples'][args.sample]}

    # plot samples 
    for name, sample_config in config['samples'].items():
        
        # set sample
        config['name'] = name
        config['sample'] = sample_config['sample']

        # set out dir
        config['out_dir'] = os.path.join(config['base_out_dir'], config['name'])
        os.makedirs(config['out_dir'], exist_ok=True)

        # make plots
        print(f"\nPlotting {name}...")
        if not config['tracks']:
            make_jet_plots(config)
        else:
            to.make_track_origin_plots(config)
            vert.make_vertexing_plots(config)

    print('\nFinished plotting.\n')


if __name__ == "__main__":
    main()
