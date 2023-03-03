"""
Functions to produce plots of the track origin performance.
"""

import os
import yaml

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from atlasify import AtlasStyle

import utils.evaluation as ev
import utils.plotting as pu
import utils.h5tools as h5t


# merge some origins to reduce the number of curves on the plots
DO_MERGE_ORIGINS = False

atlas_style = AtlasStyle(
    atlas='Simulation Internal',
    offset=7,
    indent=7,
    font_size=13,
    label_font_size=13,
    sub_font_size=10,
    enlarge=1
)

def get_node_origins(train_config_path):
    """
    Get the node origins dict from the training config file.
    """

    # load the training config
    with open(train_config_path, 'r') as fp:
        train_config = yaml.safe_load(fp)

    # get the node origins dict
    node_origins = {v['label']: v['name'] for v in train_config['node_class_dict'].values()}
    
    # remove tau origins, these are not yet included in the training
    node_origins[6] = node_origins[7]
    del(node_origins[7])

    #node_origins_inv = dict(zip(node_origins.values(), node_origins.keys()))

    return node_origins


def filter_track_origin(df, node_origins, origin, label='truthOriginLabel'):
    """
    Remove tracks with label `origin` from `df`
    """

    assert(origin == 'fromTau'), 'This is only supported for fromTau at the moment'
    
    # shouldn't actually be loaded in, but still
    df = df.drop(columns=origin, errors='ignore')
    
    # update the label
    df[label] = df[label].replace(7, 6)
    df["GN1OriginLabel"] = df["GN1OriginLabel"].replace(7, 6)
    
    # rename columsn to int 
    #df = df.rename(columns=node_origins_inv)

    return df


origin_remapping = {
    "Pileup": 2,
    "Fake": 3,
    "Primary": 1,
    "FromB": 0,
    "FromBC": 0,
    "FromC": 0,
    "OtherSecondary": 1,
}
grouped_origins = {
    "new_HF": ["FromB", "FromBC", "FromC"],
    "new_Primary": ["Primary", "OtherSecondary"],
    "new_Pileup": ["Pileup"],
    "new_Fake": ["Fake"],
}
new_origin_dict = dict(zip(range(len(grouped_origins)), grouped_origins))


def load_track_origins(config, pt_range):
    """
    Load model eval file and read track origin classification scores and labels
    """
    
    print(f" - Loading origins eval")

    models = {}
    for model, gnn_config in config['models'].items():
        
        # only have origin information in the pytorch eval files
        if gnn_config['source'] != 'pytorch':
            continue

        # we didn't train the aux tasks, skip
        train_config_path = os.path.join(gnn_config['save_dir'], gnn_config['id'], 'GN1.yaml')
        if not os.path.exists(train_config_path):
            continue

        # get the node origin dicts
        node_origins = get_node_origins(train_config_path)
        gnn_config['node_origins'] = node_origins

        # get the eval path
        eval_dir = os.path.join(gnn_config['save_dir'], gnn_config['id'])
        eval_path = ev.get_checkpoint_evaluation_path(eval_dir, config['sample'], epoch=gnn_config.get('epoch'), fname=gnn_config.get('fname'))
        
        # load info
        jet_variables = ['n_tracks', 'pt']
        track_variables = ['truthOriginLabel', 'GN1OriginLabel'] + list(map(str, node_origins.values()))
        jet_df = h5t.get_jet_df(eval_path, jet_variables, num_jets=config['num_jets'])
        jet_pt = jet_df.pt.repeat(jet_df.n_tracks)
        df = h5t.get_jet_df(eval_path, track_variables, 'nodes', num_jets=jet_df.n_tracks.sum())

        # merge info
        assert(len(jet_pt) == len(df))
        df['pt'] = jet_pt.values
        
        # apply pt cuts
        df = ev.apply_pt_cuts(df, pt_range)

        # replace tau label
        df = filter_track_origin(df, node_origins, 'fromTau')

        # do some merging of origins 
        if DO_MERGE_ORIGINS:
            origin_dict = gnn_config['node_origins']
            origin_dict_inv = {v: k for k, v in origin_dict.items()}

            remap_dict = {origin_dict_inv[origin]: origin_remapping[origin] for origin in origin_remapping}


            df['truthOriginLabel'] = df['truthOriginLabel'].map(remap_dict)
            df['GN1OriginLabel'] = df['GN1OriginLabel'].map(remap_dict)
            
            for new_origin, subs in grouped_origins.items():
                df[new_origin] = df[subs].sum(axis=1)

        models[model] = df


    return config, models 


