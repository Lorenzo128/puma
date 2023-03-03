"""
Functions to produce plots of the track origin performance.
"""

import os
import yaml
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib
import seaborn as sns
from atlasify import AtlasStyle

import utils.evaluation as ev
import utils.plotting as pu
import utils.track_origins as to
import utils.h5tools as h5t


atlas_style = AtlasStyle(
    atlas='Simulation Internal',
    offset=7,
    indent=7,
    font_size=13,
    label_font_size=13,
    sub_font_size=10,
    enlarge=1
)

model_styles = {
    'SV1': {
        'label': 'SV1',
        'color': 'series:orange',
        #'marker': 'o', 
    },
    'JF': {
        'label': 'JetFitter',
        'color': 'series:purple',
        #'marker': 'o', 
    },
    'GN1': {
        'label': 'GN1',
        'color': 'series2:blue',
        #'marker': 'o', 
    },
    'GN1 HF': {
        'label': 'GN1 HF',
        'color': 'series2:blue',
        #'marker': '^', 
    },
}

# flags for custom vertex processing
REQUIRE_GNN_HF_EFF = True
DO_GN1_REMOVE_PV = True
DO_SV1_TRUTH_INCLSUIVE = True
FAKE_REQUIRE_GNN_HF = True
FAKE_DEF_REQUIRE_TRUTH_SECONDARY = False


def load_vertexing(config, pt_range):
    """
    Load vertexing information into DataFrames
    """

    print(f" - Loading vertexing eval")

    models = {}
    for model, gnn_config in config['models'].items():
        
        # only have vertexing information in the pytorch eval files
        if gnn_config['source'] != 'pytorch':
            continue

        # we didn't train the aux tasks, skip
        train_config_path = os.path.join(gnn_config['save_dir'], gnn_config['id'], 'GN1.yaml')
        if not os.path.exists(train_config_path):
            continue

        # get the node origin dicts
        node_origins = to.get_node_origins(train_config_path)
        gnn_config['node_origins'] = node_origins

        # get the eval path
        eval_dir = os.path.join(gnn_config['save_dir'], gnn_config['id'])
        eval_path = ev.get_checkpoint_evaluation_path(eval_dir, config['sample'], epoch=gnn_config.get('epoch'), fname=gnn_config.get('fname'))

        # get jet info
        jet_vars = ['n_tracks', 'pt', 'flavour']
        jet_df = h5t.load_jet_df(eval_path, jet_vars, 'jets', num_jets=config['num_jets'])

        # get node info
        vertexing_vars = ['JFVertexIndex', 'SV1VertexIndex', 'GN1OriginLabel', 'truthOriginLabel', 'truthVertexIndex', 'GN1VertexIndex']
        df = ev.load_jet_df(eval_path, vertexing_vars, 'nodes', num_jets=jet_df.n_tracks.sum())
        
        # merge
        for jet_var in jet_vars:
            df[jet_var] = jet_df[jet_var].repeat(jet_df.n_tracks).values
        df['jet_id'] = np.repeat(np.arange(config['num_jets']), jet_df.n_tracks)

        # apply pt cuts
        df = ev.apply_pt_cuts(df, pt_range)
        models[model] = df

    return config, models 


def get_reco_pv_index(df):
    """
    Given an input DataFrame representing a single jet, determine which predicted vertex 
    has the most truth primary tracks in it and return the index of this vertex.
    """

    # overwritten in the loop
    max_n_primary_tracks = 0
    reco_pv_index = -1

    # for each predicted gnn vertex that is not PU/fake
    for reco_vtx_idx, vtx_df in df.groupby('GN1VertexIndex', sort=False):

        # skip if PU/fake
        if reco_pv_index == -2:
            continue

        # count truth primary tracks
        n_primary_tracks = (vtx_df['GN1OriginLabel'] == 2).sum()

        # if this vertices has the most primary tracks, save it
        if n_primary_tracks > max_n_primary_tracks:
            max_n_primary_tracks = n_primary_tracks
            reco_pv_index = reco_vtx_idx

    return reco_pv_index


def remove_GN1_reco_pv(df):
    """
    Given an input DataFrame representing a single jet, remove tracks in the predicted
    primary vertex.
    """
    
    # determine reco PV
    reco_pv_index = get_reco_pv_index(df)
    
    # if we have a PV, remove the tracks in it
    if reco_pv_index != -1:
        df.loc[df['GN1VertexIndex'] == reco_pv_index, 'GN1VertexIndex'] = -2
    
    return df


def get_inclusive_truth_sv(df, target_flavour=5):
    """
    Given an input DataFrame representing a single jet, put all truth HF tracks
    in the same truth vertex. Useful for more accurately representing the performance
    of SV1.
    """
    if target_flavour == 5:
        df.loc[(df.truthOriginLabel == 3) | (df.truthOriginLabel == 4), 'truthVertexIndex'] = 999
    elif target_flavour == 4:
        df.loc[df.truthOriginLabel == 5, 'truthVertexIndex'] = 999
    else:
        raise ValueError(f"unsupported flavour {target_flavour}")
    return df


def group_GN1_HF_Vertex(vertex_df, target_flavour=5):
    """
    Put different predicted GN1 heavy flavor vertices into the same GN1 reco vertex.
    """
    if target_flavour == 5:
        if ((vertex_df.GN1OriginLabel == 3) | (vertex_df.GN1OriginLabel == 4)).any():
            vertex_df['GN1VertexIndex'] = 999
    elif target_flavour == 4:
        if (vertex_df.GN1OriginLabel == 5).any():
            vertex_df['GN1VertexIndex'] = 999
    else:
        raise ValueError(f"unsupported flavour {target_flavour}")
    return vertex_df



def get_inclusive_reco_sv(jet_df, target_flavour=5):
    """
    Group reconstructed JF and GN1 vertices into a single vertex to measure
    inclusive vertex reconstruction performance.
    """
    
    # group all JF vertices
    jet_df.loc[jet_df['JFVertexIndex'] >= 0, 'JFVertexIndex'] = 999

    # group all HF GNN vertices
    jet_df = jet_df.groupby('GN1VertexIndex', sort=False).apply(group_GN1_HF_Vertex, target_flavour=target_flavour)

    return jet_df


def get_vertex_efficiency(truth_vertex_df, jet_df, model, precision_cut=1, recall_cut=1):
    """
    Given a DataFrame of tracks in a truth SV, and a DataFrame of tracks in a predicted SV, 
    compute the precision and recall of the SV prediction.
    """

    row = {'jet_id': truth_vertex_df.jet_id.iat[0], 'pt': truth_vertex_df.pt.iat[0], 'n_track_truth': len(truth_vertex_df)}
    row['n_truth_vertex'] = jet_df.truthVertexIndex.nunique()

    # for each predicted vertex (GNN/SV1/JF)
    for reco_vertex_index, reco_vertex_df in jet_df.groupby(model+'VertexIndex', sort=False):
        
        # skip if PU/fake
        if reco_vertex_index == -2:
            continue

        # calculate precision and recall
        vtx_recall = truth_vertex_df.index.isin(reco_vertex_df.index).mean()
        vtx_precision = reco_vertex_df.index.isin(truth_vertex_df.index).mean()

        # see if the predicted vertex pass finding cuts
        if vtx_recall >= recall_cut and vtx_precision >= precision_cut:
            
            # if one track vertex, has to be pure
            if len(truth_vertex_df) == 1 and vtx_precision != 1:
                continue
            
            # require GNN to predict a heavy flavour track in the vertex to call efficient
            if REQUIRE_GNN_HF_EFF and model == 'GN1':
                if not ((reco_vertex_df.GN1OriginLabel == 3) | (reco_vertex_df.GN1OriginLabel == 4) | (reco_vertex_df.GN1OriginLabel == 5)).any():
                    continue

            # return the first vertex that passes the cuts. Assuming a reasonable recall, there can only be one
            return {**row, model+'_eff': 1, model+'_precision': vtx_precision, model+'_recall': vtx_recall}

    # return inefficient vertex
    return {**row, model+'_eff': 0, model+'_precision': -1, model+'_recall': -1}


def get_effs(df, precision_cut=1, recall_cut=1, target_flavour=5, inclusive=False):
    """
    Get vertex finding efficiencies.
    """
    
    # select target jets
    df = df[df.flavour == target_flavour].copy()

    # results go here
    models = {model: [] for model in ['SV1', 'JF', 'GN1']}
    if not inclusive:
        del models['SV1']

    # loop throgugh target jets
    for jet_id, jet_df in df.groupby('jet_id', sort=False):

        # remove the GNN predicted PV
        if DO_GN1_REMOVE_PV:
            jet_df = remove_GN1_reco_pv(jet_df)        

        # setup inclusive definitions
        if inclusive:
            jet_df = get_inclusive_truth_sv(jet_df, target_flavour=target_flavour)
            jet_df = get_inclusive_reco_sv(jet_df, target_flavour=target_flavour)        
        
        # for each truth vertex
        for truth_vertex_index, truth_vertex_df in jet_df.groupby('truthVertexIndex', sort=False):

            # skip truth primary (= 0) and PU (= -2) vertices
            if truth_vertex_index <= 0:
                continue 

            # ensure we only have b decay tracks in the truth SV
            if target_flavour == 5:
                if not all((truth_vertex_df.truthOriginLabel == 3) | (truth_vertex_df.truthOriginLabel == 4)):
                    continue
            elif target_flavour == 4:
                if not all(truth_vertex_df.truthOriginLabel == 5):
                    continue
            
            if inclusive:
                models['SV1'].append(get_vertex_efficiency(truth_vertex_df, jet_df, 'SV1', precision_cut, recall_cut))
            models['JF'].append(get_vertex_efficiency(truth_vertex_df, jet_df, 'JF', precision_cut, recall_cut))
            models['GN1'].append(get_vertex_efficiency(truth_vertex_df, jet_df, 'GN1', precision_cut, recall_cut))

    return {model: pd.DataFrame(data) for model, data in models.items()}


def get_jet_fake_vertex_rate(jet_df, model, matching_cut=0.5):

    # model specific processing
    if DO_GN1_REMOVE_PV and model == 'GN1':
        jet_df = remove_GN1_reco_pv(jet_df)
    if DO_SV1_TRUTH_INCLSUIVE and model == 'SV1':
        jet_df = get_inclusive_truth_sv(jet_df)

    rows = []

    # for each found vertex
    for reco_vertex_index, reco_vertex_df in jet_df.groupby(model+'VertexIndex', sort=False):
        
        # ensure valid vertex
        if reco_vertex_index < 0:
            continue

        # loop through truth vertices
        row = {'jet_id': jet_df.jet_id.iat[0], 'pt': jet_df.pt.iat[0], 'n_track_reco': len(reco_vertex_df)}
        for truth_vertex_index, truth_vertex_df in jet_df.groupby('truthVertexIndex', sort=False):
            
            # large enough fraction of tracks in the reco vertex are found in the same truth vertex 
            if reco_vertex_df.index.isin(truth_vertex_df.index).mean() > matching_cut:

                # if one track reco vertex, ensure the truth vertex doesn't have more than one track
                if len(reco_vertex_df) == 1 and len(truth_vertex_df) != 1:
                    continue
                
                # ok, it's not a fake vertex
                row[f'{model}_isFake'] = 0
                rows.append(row)
        
        # if there is no truth vertex which contains at least `matching_cut` of the reco vertex tracks, call it fake
        if f'{model}_isFake' not in row:
            row[f'{model}_isFake'] = 1
            rows.append(row)

    return rows


def get_fake_rates(df, matching_cut=0.5):
    """
    Get vertex finding fake rates in light jets.
    """
    
    # select target jets
    df = df[df.flavour == 0].copy()

    # results go here
    models = {model: [] for model in ['SV1', 'JF', 'GN1']}

    # loop throgugh jets
    for jet_id, jet_df in df.groupby('jet_id', sort=False):

        # for each model, get the fake rate for this jet
        for model in models:
            models[model].extend(get_jet_fake_vertex_rate(jet_df, model, matching_cut))

    return {model: pd.DataFrame(data) for model, data in models.items()}


def has_no_secondaries(vertex_df, require_pred_hf=False):
    """
    Returns False if the vertex contains some tracks with truthOrigins that could be displaced.
    Also optionally returns False 

    Return True if we are going to call this a fake track.
    """

    if FAKE_DEF_REQUIRE_TRUTH_SECONDARY:
        # if there are otherSecondary tracks, might have reconstructed a real displaced decay (material interactions etc)
        if (vertex_df.truthOriginLabel == 7).any():
            return False
        
        # if there are HF tracks, might have reconstructed a real displaced decay
        if ((vertex_df.truthOriginLabel == 3) | (vertex_df.truthOriginLabel == 4) | (vertex_df.truthOriginLabel == 5)).any():
            return False

    # if there are PU/fake tracks, might have reconstructed a real displaced decay
    #if ((vertex_df.truthOriginLabel == 0) | (vertex_df.truthOriginLabel == 1)).any():
    #    return False
    
    # GN1 doesn't predict any HF track in the vertex, we can use this to reduce the fake rate
    if require_pred_hf:
    #    if (vertex_df.GN1OriginLabel == 7).any():
    #        return False
        if not ((vertex_df.GN1OriginLabel == 3) | (vertex_df.GN1OriginLabel == 4) | (vertex_df.GN1OriginLabel == 5)).any():
            return False

    # ok, we have a fake vertex
    return True


def get_num_reco_vert_without_secondary(jet_df, alg, require_pred_hf=False):
    """
    Count the number of fake vertices 
    """
    alg = alg.replace(' HF', '') ### hack for GNN HF model to access the correct vertex index
    vertices = jet_df.groupby(f'{alg}VertexIndex', sort=False).apply(has_no_secondaries, require_pred_hf=require_pred_hf)
    return vertices[vertices.index >= 0].sum()


def get_jet_fake_vertex_rate_v2(jet_df, model, matching_cut=0.5):
    """
    Rate to reconstruct a vertex which does not contain any otherSecondary
    or Heavy Flavour track. These are "true fakes" in the sense of being fake
    vertices which do not come from material interactions of other "real" displaced
    vertices in light jets.
    """
    
    # model specific processing
    if DO_GN1_REMOVE_PV and model == 'GN1':
        jet_df = remove_GN1_reco_pv(jet_df)
    if DO_SV1_TRUTH_INCLSUIVE and model == 'SV1':
        jet_df = get_inclusive_truth_sv(jet_df)

    if FAKE_REQUIRE_GNN_HF and model == 'GN1 HF':
        require_pred_hf = True
    else:
        require_pred_hf = False
    
    row = {'jet_id': jet_df.jet_id.iat[0], 'pt': jet_df.pt.iat[0], f'{model}_isFake': 0}
    row[f'{model}_isFake'] = get_num_reco_vert_without_secondary(jet_df, model, require_pred_hf=require_pred_hf)

    return row


def get_fake_rates_v2(df, matching_cut=0.5):
    """
    Get vertex finding fake rates in light jets.
    """
    
    # select target jets
    df = df[df.flavour == 0].copy()

    # results go here
    models = {model: [] for model in ['SV1', 'JF', 'GN1', 'GN1 HF']}

    # loop throgugh jets
    for jet_id, jet_df in df.groupby('jet_id', sort=False):

        # for each model, get the fake rate for this jet
        for model in models:
            models[model].append(get_jet_fake_vertex_rate_v2(jet_df, model, matching_cut))
    
    return {model: pd.DataFrame(data) for model, data in models.items()}


def plot_vertexing_eff(config, models, wp, target_flavour=5, var='eff', ntrack=None, incl=None, save_path=None):
    """
    Plot vertexing efficiency as a function of get pT.
    """

    # preselections
    for model, df in models.items():
        # if precision, require reco vertex
        if var == 'precision':
            df = df[df[f'{model}_eff'] == 1]
        # ntrack selection
        if ntrack == '1 track':
            df = df[df.n_track_truth == 1]
        if ntrack == '1+ track':
            df = df[df.n_track_truth >= 1]
        elif ntrack == '2+ track':
            df = df[df.n_track_truth >= 2]
        models[model] = df

    # plotting setup
    sample_config = config['samples'][config['sample']]
    pt_range = sample_config['pt_range']
    xb = np.linspace(pt_range[0], pt_range[1], 10)
    fig, ax = pu.get_plot()

    # plot points
    bin_width = (xb[1] - xb[0])
    x_bins = xb
    xs = (x_bins[1:] + x_bins[:-1]) / 2 # get bin midpoints
    plt_handles = []
    plt_labels = []
    for model, df in models.items():
        groups = df.groupby(pd.cut(df['pt'], x_bins))[f'{model}_{var}']
        means = groups.mean(); sems = groups.sem()
        style = model_styles[model]
        handle, label = pu.plot_step_hist(ax, xs, x_bins, means,  sems, bin_width, style)
        plt_handles.append(handle); plt_labels.append(label)
        ax.errorbar(xs, means, xerr=bin_width/2, **style, linestyle='None', fmt="none")

    # label axes
    ax.set_xlabel('Jet $p_T$ [GeV]')
    if var == 'eff':
        ax.set_ylabel('Vertex Finding Efficiency')
    elif var == 'precision':
        ax.set_ylabel('Vertex Purity')

    # format and legends
    ax.set_xlim(pt_range)
    ymin, ymax = ax.set_ylim();
    if var == 'eff':
        ax.set_ylim(0, 1.2)
    elif var == 'precision':
        ax.set_ylim(ymin, 1.2)

    desc = config['sample_str'] + f'\n${ev.flav(target_flavour)}$-jets, {ntrack} truth vertices'
    if incl == 'incl':
        desc += '\nInclusive finding'
    elif incl == 'excl':
        desc += '\nExclusive finding'
    atlas_style.apply(axes=ax, subtext=desc)
    ax.legend(handles=plt_handles, labels=plt_labels, loc='upper right', fontsize=13)

    # save plot
    if save_path:
        fig.savefig(save_path)


def plot_vertexing_fake_rate(config, models, ntrack=None, save_path=None):
    """
    Plot vertexing fake rate as a function of get pT.
    """

    for model, df in models.items():
        if ntrack == 'all':
            continue
        elif ntrack == '1 track':
            models[model] = df[df.n_track_reco == 1]
        elif ntrack == '2+ track':
            models[model] = df[df.n_track_reco >= 2]

    sample_config = config['samples'][config['sample']]
    pt_range = sample_config['pt_range']
    xb = np.linspace(pt_range[0], pt_range[1], 10)

    fig, ax = pu.get_plot()

    kwargs = {'ax': ax, 'x_bins': xb, 'fit_reg': False}
    for model, df in models.items():
        sns.regplot(x=df['pt'], y=df[f'{model}_isFake'], **kwargs, **model_styles[model])

    ax.set_xlabel('Jet $p_T$ [GeV]'); ax.set_ylabel('Vertex Finding Rate')
    ymin, ymax = ax.set_ylim(); ax.set_ylim(0, round(ymax*1.15, 1))
    desc = config['sample_str'] + f'\nLight jets'#, {ntrack} reco vertices'
    atlas_style.apply(axes=ax, subtext=desc)
    ax.legend(loc='upper right', fontsize=13)

    if save_path:
        fig.savefig(save_path)


def get_eff_fname(config, wp, ntrack, incl, target_flavour):
    sample = config['sample']
    out_dir = config['out_dir']
    fname = f"{sample}_{ev.flav(target_flavour)}jet_vert_eff_{ntrack.replace(' ', '_')}_{incl}.pdf"
    return os.path.join(out_dir, fname)


def plot_vertex_efficiency(config, models, target_flavour=5):
    """
    Wrapper script to make all the efficiency plots.
    """
    
    sample = config['sample']

    print(f" - Plot vertex finding efficiency (warning: slow!)")
    for model, gnn_config in config['models'].items():
        
        # exclusive vertexing performance, loose WP
        results = get_effs(models[model], target_flavour=target_flavour, precision_cut=0.5, recall_cut=0.65, inclusive=False)

        wp = 'loose'; ntrack = '1 track'; incl = 'excl'
        save_path = get_eff_fname(config, wp, ntrack, incl, target_flavour)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, ntrack=ntrack, incl=incl, save_path=save_path)

        wp = 'loose'; ntrack = '1+ track'; incl = 'excl'
        save_path = get_eff_fname(config, wp, ntrack, incl, target_flavour)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, ntrack=ntrack, incl=incl, save_path=save_path)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, var='precision', ntrack=ntrack, incl=incl, save_path=save_path.replace('eff', 'pur'))

        wp = 'loose'; ntrack = '2+ track'; incl = 'excl'
        save_path = get_eff_fname(config, wp, ntrack, incl, target_flavour)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, ntrack=ntrack, incl=incl, save_path=save_path)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, var='precision', ntrack=ntrack, incl=incl, save_path=save_path.replace('eff', 'pur'))

        # inclusive vertexing performance, loose WP
        results = get_effs(models[model], target_flavour=target_flavour, precision_cut=0.5, recall_cut=0.65, inclusive=True)

        wp = 'loose'; ntrack = '1 track'; incl = 'incl'
        save_path = get_eff_fname(config, wp, ntrack, incl, target_flavour)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, ntrack=ntrack, incl=incl, save_path=save_path)

        wp = 'loose'; ntrack = '1+ track'; incl = 'incl'
        save_path = get_eff_fname(config, wp, ntrack, incl, target_flavour)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, ntrack=ntrack, incl=incl, save_path=save_path)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, var='precision', ntrack=ntrack, incl=incl, save_path=save_path.replace('eff', 'pur'))

        wp = 'loose'; ntrack = '2+ track'; incl = 'incl'
        save_path = get_eff_fname(config, wp, ntrack, incl, target_flavour)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, ntrack=ntrack, incl=incl, save_path=save_path)
        plot_vertexing_eff(config, deepcopy(results), wp=wp, target_flavour=target_flavour, var='precision', ntrack=ntrack, incl=incl, save_path=save_path.replace('eff', 'pur'))


def plot_vertex_fake_rate(config, models):
    """
    Wrapper script to make all the fake rate plots.
    """

    sample = config['sample']

    print(f" - Plot vertex finding fake rate (warning: slow!)")
    for model, gnn_config in config['models'].items():

        results = get_fake_rates(models[model], matching_cut=0.5)
        
        """ntrack = '1 track'
        save_path = os.path.join(config['out_dir'], f"{sample}_vert_fake_{ntrack.replace(' ', '_')}.pdf")
        plot_vertexing_fake_rate(config, deepcopy(results), ntrack=ntrack, save_path=save_path)

        ntrack = '2+ track'
        save_path = os.path.join(config['out_dir'], f"{sample}_vert_fake_{ntrack.replace(' ', '_')}.pdf")
        plot_vertexing_fake_rate(config, deepcopy(results), ntrack=ntrack, save_path=save_path)"""

        results = get_fake_rates_v2(models[model])
        ntrack = 'all'
        save_path = os.path.join(config['out_dir'], f"{sample}_vert_fake_{ntrack.replace(' ', '_')}_v2.pdf")
        plot_vertexing_fake_rate(config, deepcopy(results), ntrack=ntrack, save_path=save_path)



def make_vertexing_plots(config):
    """
    Make vertexing performance plots.
    """

    # load info info
    sample = config['sample']
    sample_config = config['samples'][sample]
    config, models = load_vertexing(config, pt_range=sample_config['pt_range'])
    config['sample_str'] = ev.get_sample_string(sample_config)

    # make the plots
    plot_vertex_efficiency(config, models, target_flavour=5)
    #plot_vertex_efficiency(config, models, target_flavour=4)
    #plot_vertex_fake_rate(config, models)

