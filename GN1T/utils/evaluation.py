"""
    Functions used to evaluate the performance of the taggers.
"""

import glob
import os
import yaml
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import scipy

from puma.metrics import eff_err, rej_err

import utils.h5tools as h5t

# ---------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------
def get_sample_string(sample_config, com_string='$\\sqrt{s}$ = 13 TeV\n', add_str='', pt_range=None):
    """
    Get a string representation of a sample.
    """

    # get information about sample from config
    if pt_range is None:
        pt_range = sample_config['pt_range']
    sample_str = sample_config['latex']

    # form sample string
    pt_cut_str = f'$ {pt_range[0]} < p_T < {pt_range[1]} $ GeV' if pt_range is not None else None
    sample_str = ', '.join(filter(None, [sample_str, pt_cut_str, add_str]))

    sample_str = com_string + sample_str

    return sample_str


def load_config(config_path):
    
    # read config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # add model configs
    model_configs = os.path.join(os.path.dirname(config_path), config['model_configs'])
    with open(model_configs, 'r') as file:
        model_configs = yaml.safe_load(file)["models"]
    config["models"] = {model: model_configs[model] for model in config["models"]}

    return config

# ---------------------------------------------------------------------
# Loading eval samples
# ---------------------------------------------------------------------
def get_checkpoint_evaluation_path(model_eval_dir, sample, epoch=None, fname=None):
    """
    Get the evaluation sample filepath.
    """

    # training checkpoints go in a subdir
    ckpt_dir = os.path.join(model_eval_dir, 'ckpts')

    # if a filename is provided, use it directly
    if fname:
        evals = glob.glob(f'{ckpt_dir}/{fname}')

    # otherwise, look for checkpoints which contain the right sample
    else:
        evals = glob.glob(f'{ckpt_dir}/*.h5')
        evals = [f for f in evals if f'_{sample}.h5' in f]

    # if there is more than one, we can distinguish by epoch
    if not len(evals) == 1 and epoch:
        evals = [f for f in evals if f'epoch={epoch:02d}' in f ]

    # assertions
    assert (len(evals) != 0), f'No file found in {ckpt_dir}'
    assert (len(evals) <= 1), f'Too many files found in {ckpt_dir}'

    # return path
    model_eval_path = os.path.join(ckpt_dir, evals[0])
    return model_eval_path


def rename_dll_col(x):
    if '_pb' in x: return 'p_b'
    elif '_pc' in x: return 'p_c'
    elif '_pu' in x: return 'p_u'
    elif 'HadronConeExclTruthLabelID' in x: return 'flavour'
    elif 'labels' in x: return 'flavour'
    elif 'n_tracks_loose' in x: return 'n_tracks'
    else: return x


def select_pass_A_fail_B(df_A, df_B, wp_cut_A, wp_cut_B, disc='D_b'):
    pass_A = df_A[disc] > wp_cut_A
    pass_B = df_B[disc] > wp_cut_B
    return df_A.loc[pass_A & ~pass_B]


def load_umami_models(config):
    """
    Load a set of model predictions from h5 files in the format written out by the TDD.
    This includes the umami hybrid samples that are produced after running the prepare
    preprocessing stage
    """

    umami_models = {}
    for umami_model, model_config in config['models'].items():

        if model_config['source'] != 'umami':
            continue
        
        # jet variables to load
        if model_config.get('is_mv2'):
            jet_variables = ['MV2c10_discriminant']
            jet_variables += ['pt', 'eta', 'HadronConeExclTruthLabelID']
        else:
            jet_variables = [f'{umami_model}_{s}' for s in ['pb', 'pc', 'pu']]
            jet_variables += ['pt', 'eta', 'HadronConeExclTruthLabelID', 'n_tracks_loose']
        if 'load_additional_vars' in config:
            jet_variables += config['load_additional_vars']
        
        # h5 path(s) to load
        base_path = Path(model_config['save_dir'], model_config['dataset_version'])
        subdirs = [x for x in base_path.iterdir() if x.is_dir()]
        if 'hybrids' in [x.stem for x in subdirs]:
            h5_paths = [base_path / 'hybrids' / f"MC16d-inclusive_testing_{config['sample']}_PFlow.h5"]
        else:
            sample_id = {'ttbar': ['410470', '600012'], 'zprime': ['800030'],}
            for subdir in subdirs:
                if any(s in subdir.stem for s in sample_id[config['sample']]) or config['sample'] in subdir.stem:
                    h5_paths = glob.glob(f'{subdir}/*.h5')

        # load jets 
        df = pd.DataFrame()
        n_files = 0
        for h5_path in h5_paths:
            df_ = h5t.load_jets(h5_path, jet_variables, num_jets=config['num_jets'], rescale_pt=True)
            print(f'\tgot {len(df_)} jets from {h5_path}')
            df = pd.concat([df, df_], ignore_index=True)
            n_files += 1
            if config['num_jets'] > 0 and len(df) > config['num_jets']:
                df = df.iloc[:config['num_jets']]
                break
        print(f'\t{umami_model}: got a total of {len(df)} jets from {n_files} files')
        print(f'\t\tbjets: {len(df.loc[df.flavour == 5])}')
        print(f'\t\tcjets: {len(df.loc[df.flavour == 4])}')
        print(f'\t\tljets: {len(df.loc[df.flavour == 0])}')
        print(f'\t\ttaujets: {len(df.loc[df.flavour == 15])}')
        
        umami_models[umami_model] = df
 
    return umami_models 


def load_pytorch_models(config):
    """
    Load a set of GNN model predictions from previously produced eval files.
    """

    gnn_models = {}
    for gnn_model, model_config in config['models'].items():

        if model_config['source'] != 'pytorch':
            continue
        
        # get path
        eval_dir = os.path.join(model_config['save_dir'], model_config['id'])
        gnn_eval_path = get_checkpoint_evaluation_path(
            eval_dir,
            config["sample"],
            epoch=model_config.get("epoch"),
            fname=model_config.get("fname"),
        )

        # get list of variables we need to loda
        jet_variables = ['salt_pb', 'salt_pc', 'salt_pu', 'pt', 'HadronConeExclTruthLabelID', 'eta']
        if model_config['ntrack_light']:
            jet_variables += ['n_tracks']

        # load jets
        gnn_df = h5t.load_jets(gnn_eval_path, jet_variables, rescale_pt=True, num_jets=config['num_jets'])
        
        gnn_models[gnn_model] = gnn_df

    return gnn_models 


def load_models(config, pt_range=None, absEta_range=None):
    """
    Load all models in the plotting config into pandas DataFrames.
    """

    print(f" - Loading models")

    # combine models
    models = {
        **load_umami_models(config),
        **load_pytorch_models(config)
    }

    # apply selections
    if pt_range is None:
        pt_range = config['samples'][config['name']]['pt_range']
    if absEta_range is None and 'absEta_range' in config['samples'][config['name']]:
        absEta_range = config['samples'][config['name']]['absEta_range']
    models = filter_all(models, pt_range, absEta_range)

    # add scores
    models = add_scores(config, models)

    # get info string for plot
    config['sample_str'] = get_sample_string(config['samples'][config['name']], pt_range=pt_range)

    return config, models
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------
def apply_pt_cuts(df, pt_range, col='pt'):
    """
    Remove jets falling outside the pt selection
    """

    if pt_range is None: return models
    assert(len(df) > 0), 'No input jets.'
    if col == 'eta':
        df = df.loc[(abs(df[col]) > pt_range[0]) & (abs(df[col]) < pt_range[1])]
    else:
        df = df.loc[(df[col] > pt_range[0]) & (df[col] < pt_range[1])]
    assert(len(df) > 0), f'No jets in range {pt_range}'
    return df


def filter_all(models, pt_range, absEta_range):
    """
    Apply selections to all models
    """

    if pt_range is None: return models
    models = {k: apply_pt_cuts(model, pt_range) for k, model in models.items()}
    if absEta_range is None: return models
    models = {k: apply_pt_cuts(model, absEta_range, col = 'eta') for k, model in models.items()}
    return models


def remove_jets_1trk(df):
    if 'n_tracks' not in df:
        raise ValueError('n_tracks not in df!')
    return df.loc[df['n_tracks'] > 1]
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Adding info
# ---------------------------------------------------------------------
def btag_discriminant(pb, pc, pu, fc=0.018):
    return np.log( ( pb + 1e-10 ) / ( (1.0 - fc)*pu + fc*pc + 1e-10 ) )


def ctag_discriminant(pb, pc, pu, fb=0.2):
    return np.log( ( pc + 1e-10 ) / ( (1.0 - fb)*pu  + fb*pb + 1e-10 ) )


def add_tag_discs(df, fc=0.018, fb=0.2, ntrack_light=False, is_mv2=False):
    """
    Add b-tagging and c-tagging discriminants to the model dataframe.
    Also call jets with zero or one track light if ntrack_light is True
    """

    # just use the ready-made discriminant for MV2
    if is_mv2:
        df = df.rename({'MV2c10_discriminant' : 'D_b'}, axis='columns')
        return df

    df['D_b'] = btag_discriminant(df.p_b, df.p_c, df.p_u, fc=fc)
    df['D_c'] = ctag_discriminant(df.p_b, df.p_c, df.p_u, fb=fb)
    
    # hack to call jets with ntrack <= 1 light
    if ntrack_light:
        df.loc[df.n_tracks <= 1, 'D_b'] = -999
        df.loc[df.n_tracks <= 1, 'D_c'] = -999

    return df


def get_fc_from_pt(pt, a=-6.499, b=0.0027, c=0.3627):
    return c / (1 - a * np.exp( -b * pt ) )


def get_fx(model_config, fx, df):

    if model_config[fx] == 'auto':
        target_flavour = 5 if fx == 'fc' else 4
        eff = 0.7 if fx == 'fc' else 0.2
        fx_space = get_fc_space()
        fx_scan = get_fc_scan(df, model_config, target_flavour=target_flavour, eff=eff, fc_space=fx_space)
        out = get_optimal_fc(fx_scan, fc_space=fx_space)[1]
        if fx == 'fc':
            model_config['style']['label'] = model_config['style']['label'].split(')')[0] + f') {out:.2f}'
    elif model_config[fx] == 'auto_pt':
        if fx == 'fb':
            raise ValueError()
        out = get_fc_from_pt(df.pt)
    else:
        out = float(model_config[fx])
    
    return out


def add_scores(config, models):
    """
    Add tagger disciminants for each model.
    """
    for model, df in models.items():

        # get the model config
        model_config = config['models'][model]

        # get the fc to use for the discriminant calculation
        fc = get_fx(model_config, 'fc', df)
        fb = get_fx(model_config, 'fb', df)

        # calculate the tagger discriminant
        do_ntrack_light = model_config['ntrack_light']
        models[model] = add_tag_discs(models[model], fc=fc, fb=fb, ntrack_light=do_ntrack_light, is_mv2=model_config.get('is_mv2'))
    
    return models
# ---------------------------------------------------------------------
# General Analysis
# ---------------------------------------------------------------------
def get_threshold(y_preds, eff):
    """
    Get discriminant threshold corresponding to a given efficiency.
    y_preds are predictions for jets in target class,
    e.g. b-jet predictions if considering a b-jet classifier
    """

    cut_value = np.quantile( y_preds, 1.0 - eff )
    return cut_value


def get_eff(y_preds, threshold):
    """
    Get the fraction of jets in y_preds passing a threshold cut.
    """
    return np.mean(y_preds >= threshold)


def flav(flavour=None, light=False):
    """
    Basic utility function: provide a flav as int or str and get 
    the corresponding str or int, e.g. 5 -> b

    Setting light = True will return "Light" instead of "l"
    """

    flav_dict = {'b': 5, 'c': 4, 'l': 0}
    flav_dict_inv = {v: k for k, v in flav_dict.items()}

    if isinstance(flavour, str):
        return flav_dict[flavour]

    if isinstance(flavour, int):
        if flavour == 0 and light:
            return "Light"
        return flav_dict_inv[flavour]

    raise ValueError(f"Unsupported flavour {flavour}")


def flav_rej(flavour=None, return_same=True):

    rej_dict = {'b': 'c', 'c': 'b'}

    if isinstance(flavour, str):
        if return_same:
            return rej_dict[flavour]
        else:
            return flav(rej_dict[flavour])

    if isinstance(flavour, int):
        flavour = flav(flavour)
        if return_same:
            return flav(rej_dict[flavour])
        else:
            return rej_dict[flavour]

    raise ValueError(f"Unsupported flavour {flavour}")


def flav_bkg(flavour=None, return_same=True):

    bkg_dict = {'c': 'l', 'l': 'c'}

    if isinstance(flavour, str):
        if return_same:
            return bkg_dict[flavour]
        else:
            return flav(bkg_dict[flavour])

    if isinstance(flavour, int):
        flavour = flav(flavour)
        if return_same:
            return flav(bkg_dict[flavour])
        else:
            return bkg_dict[flavour]

    raise ValueError(f"Unsupported flavour {flavour}")


def get_disc_variable(flavour):
    if flavour != 4 and flavour != 5:
        raise ValueError("Only b- and c-tagging is supported")
    return f'D_{flav(flavour)}'


def get_wp_cut_from_df(
    df,
    eff : float,
    target_flavour : int,
    eff_flavour : int = None,
    ):
    disc = get_disc_variable(target_flavour)

    # infer cut flavour
    if eff_flavour is None and df.flavour.nunique() != 1:
        eff_flavour = target_flavour
    # assume correct flavour already selected
    if eff_flavour is None:
        wp_cut = get_threshold(df[disc], eff=eff)
    # select the cut flavour
    else:
        wp_cut = get_threshold(df.loc[df.flavour == eff_flavour, disc], eff=eff)
    return wp_cut


def get_eff_from_df(
    df,
    wp_cut : float,
    target_flavour : int,
    eff_flavour : int = None,
    return_rej : bool = False,
    ):
    disc = get_disc_variable(target_flavour)

    if eff_flavour is None and df.flavour.nunique() != 1:
        eff_flavour = target_flavour

    if return_rej:
        eff_or_rej_func = lambda x, y : 1/get_eff(x, y)
        error_func = rej_err
    else:
        eff_or_rej_func = get_eff
        error_func = eff_err

    # calculate efficiency
    if eff_flavour is None:
        x = df
    else:
        x = df.loc[df.flavour == eff_flavour]

    eff_or_rej = eff_or_rej_func(x[disc], wp_cut)
    if not np.isfinite(eff_or_rej):
        raise ValueError("eff or rej -> inf, use more jets")
    eff_or_rej_error = error_func(eff_or_rej, len(x))
    
    return eff_or_rej, eff_or_rej_error


def get_wp_then_eff(
    df,
    eff : float,
    target_flavour : int,
    wp_flavour : int = None,
    eff_flavour : int = None,
    return_rej : bool = False,
    ):
    if wp_flavour is None:
        wp_flavour = eff_flavour
    wp_cut = get_wp_cut_from_df(df, eff, target_flavour, wp_flavour)
    eff, err = get_eff_from_df(df, wp_cut, target_flavour, eff_flavour, return_rej)
    return eff, err


def get_ratio_err(ys, yerr, ratio=None, binned=False):
    
    relative_err = yerr / ys
    
    if ratio is not None:
        relative_err *= ratio
        down = ratio - relative_err; up = ratio + relative_err
    else:
        down = 1 - relative_err; up = 1 + relative_err

    if binned:
        down = np.concatenate((down[:1], down))
        up = np.concatenate((up[:1], up))

    return down, up
# ---------------------------------------------------------------------
# Plot specific
# ---------------------------------------------------------------------
def get_roc_rejections(df, target_flavour=5, efficiencies=np.linspace(0.5, 1.0, 100)):
    """
    Get rejections for a model to make ROC curves
    """
    
    bc_rej_flav = flav_rej(target_flavour)
    bc_eff_str = flav(target_flavour)
    bc_rej_str = flav_rej(target_flavour, return_same=False)
    
    target_df = df.loc[df.flavour == target_flavour]
    bc_df = df.loc[df.flavour == bc_rej_flav]
    u_df = df.loc[df.flavour == 0]

    results = []
    for eff in efficiencies:
        wp_cut = get_wp_cut_from_df(target_df, eff=eff, target_flavour=target_flavour)
        bc_rej, bc_err = get_eff_from_df(bc_df,
                    target_flavour=target_flavour,
                    wp_cut=wp_cut,
                    return_rej=True
                    )
        
        u_rej, u_err = get_eff_from_df(u_df,
                    target_flavour=target_flavour,
                    wp_cut=wp_cut,
                    return_rej=True
                    )
        results.append((bc_rej, bc_err, u_rej, u_err))

    out_df = pd.DataFrame(results, columns=[f'{bc_rej_str}_rej', f'{bc_rej_str}_err', 'l_rej', 'l_err'])
    out_df[f'{bc_eff_str}_eff'] = efficiencies
    return out_df


def get_eff_at_fc(df, target_flavour=5, fc=0.018, eff=None, wp_cut=None, ntrack_light=False):
    """
    Get efficiencies for a given WP and value of fc.
    """

    # get new discriminants
    disc = get_disc_variable(target_flavour)
    if target_flavour == 5:
        df[disc] = btag_discriminant(df.p_b, df.p_c, df.p_u, fc=fc)
    elif target_flavour == 4:
        df[disc] = ctag_discriminant(df.p_b, df.p_c, df.p_u, fb=fc)
    else:
        raise ValueError("Unsupported target flavour")

    # hack to call jets with ntrack <= 1 light
    if ntrack_light:
        df.loc[df.n_tracks <= 1, disc] = -999

    # get cut
    if wp_cut is None:
        wp_cut = get_wp_cut_from_df(df, eff, target_flavour)

    # get efficiencies
    bc_rej_flav = flav_rej(target_flavour)
    bc_eff, _ = get_eff_from_df(df, wp_cut, target_flavour, bc_rej_flav)
    l_eff,  _ = get_eff_from_df(df, wp_cut, target_flavour, 0)

    return bc_eff, l_eff


def get_fc_space():
    resolution = 100
    return np.concatenate((np.logspace(-3, -1, resolution//2), np.linspace(0.1, 1.0, resolution//2)))


def get_fc_scan(df, model_config, target_flavour=5, eff=0.7, fc_space=get_fc_space(), wp_cut=None):
    """
    Loop over values in fc_space and calculate efficiencies for each value.
    """

    # work on a copy since we overwrite the discriminants
    df = df.copy()

    # loop over different values of fc
    fc_scan = []
    for fc in fc_space:

        # use pre-defined cuts?
        if isinstance(wp_cut, dict):
            this_wp_cut = wp_cut[fc]
        else:
            this_wp_cut = wp_cut
        
        # add efficiencies
        fc_scan.append(
            get_eff_at_fc(
                df, target_flavour=target_flavour, 
                fc=fc, eff=eff, wp_cut=this_wp_cut, 
                ntrack_light=model_config['ntrack_light']
            )
        )
    bc_rej_str = 'c' if target_flavour == 5 else 'b'
    return pd.DataFrame(fc_scan, columns=[f'{bc_rej_str}_eff', 'l_eff'])


def get_fc_scans(config, models, target_flavour=5, eff=0.7, fc_space=get_fc_space(), wp_cut=None):
    """
    Compute scans through different values of fc for each model. For each value of fc, calculate the 
    efficiencies of the background flavours for a fixed signal efficiency.
    """

    fc_scans = {}

    # compute a scan for each model
    for model, df in models.items():
        mc = config['models'][model]
        fc_scans[model] = get_fc_scan(
            df, mc, target_flavour=target_flavour, eff=eff, fc_space=fc_space, wp_cut=wp_cut
        )

    return fc_scans


def get_optimal_fc(fc_scan, fc_space):
    """
    After calculating an fc scan using `get_fc_scan`, find the
    optimal value of fc.
    """
    
    # normalise x- and y-axes
    xs, ys = fc_scan.values[:,0], fc_scan.values[:,1]
    xs = xs / max(xs)
    ys = ys / max(ys)
    
    # get minimum distance to origin
    min_idx = np.argmin(xs**2 + ys**2)    
    return min_idx, fc_space[min_idx]


def get_effs_by_var(
    config,
    models,
    x_bins,
    x_var,
    target_flavour,
    eff_flavour,
    eff=None,
    flat_per_bin=False,
    wp=None,
    wp_cut=None,
    ):
    if wp and (eff or flat_per_bin):
        raise ValueError("Can't use wp with eff or flat_per_bin=True")
    if wp and wp_cut:
        raise ValueError("Either provide a wp in percent or directly provide the cut")

    # for each model
    effs_by_var = {}
    kwargs = {'target_flavour': target_flavour, 'eff_flavour': target_flavour}

    for model, df in models.items():
        
        # bin in the x-variable
        groups = df.groupby(pd.cut(df[x_var], x_bins))
        
        # global calculation, same wp cut point used for all bins
        if not flat_per_bin:
            if wp:
                cut = config['models'][model]['wps'][wp]
                effs = groups.apply(get_eff_from_df, wp_cut=cut, target_flavour=target_flavour, eff_flavour=eff_flavour)
                #elif wp_cut == True:
                #    this_wp_cut = config['models'][model]['wp_cut']
                #    effs = groups.apply(get_eff_from_df, wp_cut=this_wp_cut, target_flavour=target_flavour, eff_flavour=eff_flavour)
            elif wp_cut:
                effs = groups.apply(get_eff_from_df, wp_cut=wp_cut, target_flavour=target_flavour, eff_flavour=eff_flavour)
            else:
                # calculate our own wp, eff_flavour is the efficinecy of the flavour used 
                # to calculate the wp, y-axis efficiency assumed to be target_flavour
                this_wp_cut = get_wp_cut_from_df(df, eff, target_flavour, eff_flavour)
                effs = groups.apply(get_eff_from_df, wp_cut=this_wp_cut, target_flavour=target_flavour, eff_flavour=target_flavour)
        # flat per bin - calculate a new wp cut for each bin
        else:
            effs = groups.apply(get_wp_then_eff, eff=eff, **kwargs, wp_flavour=eff_flavour)            

        df = pd.DataFrame(effs.tolist(), columns=['eff', 'err'])
        effs_by_var[model] = df
        
    return effs_by_var
# ---------------------------------------------------------------------
