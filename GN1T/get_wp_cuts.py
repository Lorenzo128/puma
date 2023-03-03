"""
Script to find cut values for a given WP

Usage:

    python get_wp_cuts.py --config configs/config.yaml -e 60 70 77 85 --sample ttbar --print_yaml
"""

import os
import yaml
import argparse
from datetime import datetime

import utils.evaluation as ev

from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers, get_good_linestyles, global_config


def parse_args():
    """
    Argument parser for the plotting script.
    """
    parser = argparse.ArgumentParser(description="Script to calculate wp cuts or efficiencies.")

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help='Path to plotting config file.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-e",
        "--effs",
        nargs='+',
        type=float,
        help='One or more efficiency value to .'
    )
    group.add_argument(
        "-w",
        "--wp_cuts",
        nargs='+',
        type=float,
        help='One or more efficiency value to .'
    )
    parser.add_argument(
        "-f",
        "--flavour",
        default="b",
        choices=["b", "c"],
        type=str,
        help='Calculate b- or c-tagging WPs.'
    )
    parser.add_argument(
        "--sample",
        required=False,
        type=str, 
        choices=['ttbar', 'zprime'], 
        help='Make plots for only ttbar or zprime, instead of both (default).'
    )
    parser.add_argument(
        "-r",
        "--rejection",
        action="store_true", 
        help="Also calculate background rejections for the specified efficiencies."
    )
    parser.add_argument(
        "--pipe_to_zprime",
        action="store_true", 
        help="Calculate WPs on ttbar, find corresponding zprime effs."
    )
    parser.add_argument(
        "--print_yaml",
        action="store_true", 
        help="Print YAML structure to add to plot config."
    )
    parser.add_argument(
        "-n",
        "--num_jets",
        required=False,
        type=float,
        help='Use this many jets to calculate WPs.'
    )

    args = parser.parse_args()
    return args


def get_wp_cuts_or_effs(config, target_flavour, effs=None, wp_cuts=None, print_yaml=False, rejection=False):
    
    outs = {}
    
    # load info info
    config, models = ev.load_models(config)

    target_flavour_int = ev.flav(target_flavour)

    print(f"Calculating cut values/effs for {target_flavour}-efficiency WPs using {config['name']}:")
    for model, df in models.items():
        
        model_config = config['models'][model]
        this_outs = []

        if effs:
            print(f'Getting cuts for {model}:')
            for eff in effs:
                wp_cut = ev.get_wp_cut_from_df(df, eff/100, target_flavour_int)
                this_outs.append(wp_cut)
                print(f' -> cut value @ {eff}% {wp_cut: .3f}')

                if rejection:
                    rejection_flavour = ev.flav_rej(target_flavour_int)
                    crej = ev.get_eff_from_df(df, wp_cut, target_flavour=target_flavour_int, eff_flavour=rejection_flavour, return_rej=True)[0]
                    lrej = ev.get_eff_from_df(df, wp_cut, target_flavour=target_flavour_int, eff_flavour=0, return_rej=True)[0]
                    print(f' ---> {ev.flav_rej(target_flavour_int, return_same=False)}-rej = {crej: .8f}')
                    print(f' ---> l-rej = {lrej: .8f}')

        if wp_cuts:
            print(f'Getting effs for {model}:')
            if isinstance(wp_cuts, dict):
                this_wp_cuts = wp_cuts[model] 
            else:
                this_wp_cuts = wp_cuts[model] 
            for wp_cut in this_wp_cuts:
                eff = ev.get_eff_from_df(df, wp_cut, target_flavour_int)[0]
                this_outs.append(eff)
                print(f' -> cut value = {wp_cut: .3f}, eff = {eff:.3f}')

        outs[model] = this_outs

    # print yaml config fragments
    if effs and print_yaml:
        print('\n\nAdd this yaml to your plot config:')
        for model, wp_cuts in outs.items():
            print(f'\n  {model}:')
            print('    wps:')
            for i, eff in enumerate(effs):
                print(f'      {eff:.0f}: {wp_cuts[i]:.3f}')

    return outs


def main():
    # parse args
    args = parse_args() 

    # load config
    config = ev.load_config(args.config)

    # add args to config
    config['num_jets']  = int(args.num_jets) if args.num_jets is not None else -1

    # make plots for a single sample
    if args.sample is not None and not args.pipe_to_zprime:
        config['samples'] = {args.sample: config['samples'][args.sample]}

    # compute effs or wps 
    for name, sample_config in config['samples'].items():
        
        if args.pipe_to_zprime and name == "zprime":
            continue

        # WPs are calculated inclusively in ttbar
        if name == 'ttbar':
            sample_config['pt_range'] = None

        # set sample
        config['name'] = name
        config['sample'] = sample_config['sample']

        outs = get_wp_cuts_or_effs(config, args.flavour, args.effs, args.wp_cuts, print_yaml=args.print_yaml, rejection=args.rejection)

    # compute zprime effs corresponding to ttbar wps
    if args.pipe_to_zprime:

        config['name'] = 'zprime'
        config['sample'] = config['samples']['zprime']['sample']

        outs = get_wp_cuts_or_effs(config, args.flavour, wp_cuts=outs)

    print('\nFinished.\n')


if __name__ == "__main__":
    main()
