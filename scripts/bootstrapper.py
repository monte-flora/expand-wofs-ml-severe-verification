from metrics import norm_aupdc, brier_skill_score, bss_reliability
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score, precision_recall_curve
import numpy as np 
import xarray as xr 
from collections import OrderedDict
from itertools import compress
from scipy.stats import binned_statistic_2d
import itertools
import pandas as pd

from joblib import Parallel, delayed 

def get_repeatable_random_states(n_bootstrap):
    base_random_state = np.random.RandomState(22)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)

    return random_num_set, base_random_state 


def digitize(data, bins):
    return np.clip(np.digitize(data, bins, right=True) - 1, 0, None)


def data_iterator(to_be_stratified, 
                  iterator, 
                  predictions, 
                  X,
                  y, 
                  known_skew,
                  n_bootstrap, 
                  forecast_time_indices=None,
                  binning=False,
                 ):
    """
    Script to simplify different iterations.
    
    Parameters:
    ------------------
        to_be_stratified, 1d array of shape (n_examples,)
            This is correspond column in the testing dataframe that will 
            be stratified. 
        iterator, an iterator of values to stratify by 
        
        #Future work 
        is_grid, indicating whether iterator is based on a grid e.g., 
            CAPE ranges or CAPE/Shear ranges.
    """
    random_num_set, base_random_state = get_repeatable_random_states(n_bootstrap)
    even_or_odd = base_random_state.choice([2, 3], size=n_bootstrap)
    
    metric_mapper = OrderedDict({'naupdc':norm_aupdc, 
                             'auc': roc_auc_score, 
                             'bss':brier_skill_score, 
                             'reliability':bss_reliability,
                             'aupdc' : average_precision_score,   
                                })

    if binning:
        to_be_stratified = digitize(to_be_stratified, bins=iterator)    
        iterator = np.unique(to_be_stratified)
        
    size = len(list(iterator))
    
    # Initialize empty datasets. 
    results = [{key : (['n_groups', 'n_boot'], np.zeros((size, n_bootstrap))) for key in metric_mapper.keys()}
               for i in range(len(predictions))]
    
    # Think of other variables to store, which will help evaluate 
    # the performance results. For example, here we have the sample counts, the skew, and day count per bin. 
    # Can we think of anymore? 
    counts = np.zeros((size, n_bootstrap))
    skew = np.zeros((size, n_bootstrap))
    day_counts = np.zeros((size, n_bootstrap))
    
    additional_vars = [counts, skew, day_counts]
    additional_var_names = ['Counts', 'Skew', 'Day Counts',]
    
    total_num_of_days = len(np.unique(X['Run Date']))

    #parallel = Parallel(n_jobs=10)
    
    if isinstance(iterator[0], list):
        df = pd.DataFrame({'X' : to_be_stratified})
    
    for i,item in enumerate(iterator):
        if isinstance(item, list):
            ###where_is_item = np.concatenate([np.where(to_be_stratified==sub)[0] for sub in item])
            where_is_item = df['X'].isin(item).values
            where_is_item  = np.where(where_is_item==True)[0]
        else:
            where_is_item  = np.where(to_be_stratified==item)[0]

        for n in range(n_bootstrap):
            # For each bootstrap resample, only sample from some subset of time steps 
            # to reduce autocorrelation effects. Not perfect, but will improve variance assessment. 
            new_random_state = np.random.RandomState(random_num_set[n])
            
            if forecast_time_indices is not None:
                # val with either be a 2, 3, or 4 and then only resample from those samples. 
                val = even_or_odd[n]
                # Find those forecast time indices that are even or odd 
                where_is_fti = np.where(forecast_time_indices%val==0)[0]
                # Find the "where_is_item" has the same indices as "where_is_fti"
                idxs_subset = list(set(where_is_item).intersection(where_is_fti))
                # Resample idxs_subset for each iteration 
                these_idxs = new_random_state.choice(idxs_subset, size=len(idxs_subset),)
            else:
                these_idxs = new_random_state.choice(where_is_item, size=len(where_is_item),)
            
            counts[i,n] = len(these_idxs)
            skew[i,n] = np.mean(y[these_idxs]) if len(these_idxs) > 0 else 0.0
            day_counts[i,n] = len(np.unique(X['Run Date'].iloc[these_idxs])) / total_num_of_days
 
            if len(these_idxs) > 0 and np.mean(y[these_idxs])>0.0:
                for metric in metric_mapper.keys():
                    for g, pred in enumerate(predictions):
                        results[g][metric][1][i,n] = metric_mapper[metric](y[these_idxs], pred[these_idxs])
    
    for r in results:
        for name, array in zip(additional_var_names, additional_vars):
            r[name] = (['counts', 'n_boot'], array)
        
    datasets = [xr.Dataset(r) for r in results]
    
    return datasets

def __bootstrapper(i, n, where_is_item, random_num_set, counts, skew, 
                   day_counts, metric_mapper, known_skew, forecast_time_indices):
    global ml_results
    global bl_results

    # For each bootstrap resample, only sample from some subset of time steps 
    # to reduce autocorrelation effects. Not perfect, but will improve variance assessment. 
    new_random_state = np.random.RandomState(random_num_set[n])
            
    if forecast_time_indices is not None:
        # val with either be a 2, 3, or 4 and then only resample from those samples. 
        val = even_or_odd[n]
        # Find those forecast time indices that are even or odd 
        where_is_fti = np.where(forecast_time_indices%val==0)[0]
        # Find the "where_is_item" has the same indices as "where_is_fti"
        idxs_subset = list(set(where_is_item).intersection(where_is_fti))
        # Resample idxs_subset for each iteration 
        these_idxs = new_random_state.choice(idxs_subset, size=len(idxs_subset),)
    else:
        these_idxs = new_random_state.choice(where_is_item, size=len(where_is_item),)
                
    counts[i,n] = len(these_idxs)
    skew[i,n] = np.mean(y[these_idxs]) if len(these_idxs) > 0 else 0.0
    day_counts[i,n] = len(np.unique(X['Run Date'].iloc[these_idxs])) / total_num_of_days
 
    if len(these_idxs) > 0 and np.mean(y[these_idxs])>0.0:
        for metric in metric_mapper.keys():
            if metric =='naupdc':
                ml_results[metric][1][i,n] = metric_mapper[metric](y[these_idxs], 
                                                                ml_predictions[these_idxs], known_skew=known_skew)
                bl_results[metric][1][i,n] = metric_mapper[metric](y[these_idxs], 
                                                                bl_predictions[these_idxs], known_skew=known_skew)
            else:
                ml_results[metric][1][i,n] = metric_mapper[metric](y[these_idxs], ml_predictions[these_idxs])
                bl_results[metric][1][i,n] = metric_mapper[metric](y[these_idxs], bl_predictions[these_idxs])
    
    
def digitize(data, bins):
    return np.clip(np.digitize(data, bins, right=True) - 1, 0, None)

def data_iterator_2d(to_be_stratified_x,
                     to_be_stratified_y,
                  ml_predictions, 
                  bl_predictions,
                  X,
                  y, 
                  known_skew,
                  n_bootstrap,
                     n_bins=2,
                  iterator_x=None,
                  iterator_y=None,   
                  forecast_time_indices=None, ):
    """
    Script to simplify different iterations.
    
    Parameters:
    ------------------
        to_be_stratified, 1d array of shape (n_examples,)
            This is correspond column in the testing dataframe that will 
            be stratified. 
        iterator, an iterator of values to stratify by 
        
        #Future work 
        is_grid, indicating whether iterator is based on a grid e.g., 
            CAPE ranges or CAPE/Shear ranges.
    """
    base_random_state = np.random.RandomState(22)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)
    even_or_odd = base_random_state.choice([2, 3], size=n_bootstrap)
    
    metric_mapper = OrderedDict({'naupdc':norm_aupdc, 
                             'auc': roc_auc_score, 
                             'bss':brier_skill_score, 
                             'reliability':bss_reliability,
                             'aupdc' : average_precision_score,   
                                })

    if iterator_x is None:
        iterator_x = np.unique(
                np.percentile(
                    to_be_stratified_x,
                    np.linspace(0, 100., n_bins),
                    interpolation="lower",
                )
            )
    if iterator_y is None:
        iterator_y = np.unique(
                np.percentile(
                    to_be_stratified_y,
                    np.linspace(0, 100., n_bins),
                    interpolation="lower",
                )
            )
    
    # Get the bin indices based on the grid edges defined above. 
    #x_bin_indices = digitize(to_be_stratified_x, bins=iterator_x)
    #y_bin_indices = digitize(to_be_stratified_y, bins=iterator_y)
    
    #combined_indices = list(zip(x_bin_indices, y_bin_indices))
    
    _,_,_,indices = binned_statistic_2d(to_be_stratified_x, to_be_stratified_y, None, 'count', 
                                    bins=[iterator_x, iterator_y], expand_binnumbers=True)
    x_bin_indices = indices[1,:]#-1
    y_bin_indices = indices[0,:]#-1
    
    size_x = np.max(x_bin_indices)#len(iterator_x)
    size_y = np.max(y_bin_indices) #len(iterator_y)
    
    ml_results = {key : (['n_groups_y', 'n_groups_x', 'n_boot'],
                         np.zeros((size_x, size_y, n_bootstrap))) for key in metric_mapper.keys()}
    
    bl_results = {key : (['n_groups_y', 'n_groups_x','n_boot'], 
                         np.zeros((size_x, size_y, n_bootstrap))) for key in metric_mapper.keys()}
    
    # Think of other variables to store, which will help evaluate 
    # the performance results. For example, here we have the sample counts, the skew, and day count per bin. 
    # Can we think of anymore? 
    counts = np.zeros((size_x, size_y, n_bootstrap))
    skew = np.zeros((size_x, size_y, n_bootstrap))
    day_counts = np.zeros((size_x, size_y, n_bootstrap))
    
    additional_vars = [counts, skew, day_counts]
    additional_var_names = ['Counts', 'Skew', 'Day Counts',]
    
    total_num_of_days = len(np.unique(X['Run Date']))

    #potential_indices = list(set(combined_indices))
    for i,j in itertools.product(np.unique(y_bin_indices), np.unique(x_bin_indices)):
        where_is_item = np.where((x_bin_indices==i) & (y_bin_indices==j))[0]
        
        i-=1; j-=1

        ###print(f'{i=}', f'{j=}', len(where_is_item))
        
        for n in range(n_bootstrap):
            # For each bootstrap resample, only sample from some subset of time steps 
            # to reduce autocorrelation effects. Not perfect, but will improve variance assessment. 
            new_random_state = np.random.RandomState(random_num_set[n])
            
            if forecast_time_indices is not None:
                # val with either be a 2, 3, or 4 and then only resample from those samples. 
                val = even_or_odd[n]
                # Find those forecast time indices that are even or odd 
                where_is_fti = np.where(forecast_time_indices%val==0)[0]
                # Find the "where_is_item" has the same indices as "where_is_fti"
                idxs_subset = list(set(where_is_item).intersection(where_is_fti))
                # Resample idxs_subset for each iteration 
                these_idxs = new_random_state.choice(idxs_subset, size=len(idxs_subset),)
            else:
                these_idxs = new_random_state.choice(where_is_item, size=len(where_is_item),)
            
            counts[j,i,n] = len(these_idxs)
            skew[j,i, n] = np.mean(y[these_idxs]) if len(these_idxs) > 0 else np.nan
            day_counts[j,i,n] = len(np.unique(X['Run Date'].iloc[these_idxs])) / total_num_of_days
 
            if len(these_idxs) > 0 and np.mean(y[these_idxs])>0.0:
                for metric in metric_mapper.keys():
                    if metric =='naupdc':
                        ml_results[metric][1][i,j,n] = metric_mapper[metric](y[these_idxs], 
                                                                ml_predictions[these_idxs], known_skew=known_skew)
                        
                        bl_results[metric][1][i,j,n] = metric_mapper[metric](y[these_idxs], 
                                                                bl_predictions[these_idxs], known_skew=known_skew)
                    else:
                        ml_results[metric][1][i,j, n] = metric_mapper[metric](y[these_idxs], ml_predictions[these_idxs])
                        bl_results[metric][1][i,j, n] = metric_mapper[metric](y[these_idxs], bl_predictions[these_idxs])
    
    for r in [ml_results, bl_results]:
        for name, array in zip(additional_var_names, additional_vars):
            r[name] = (['n_groups_y', 'n_groups_x', 'n_boot'], array)
        
    ml_ds = xr.Dataset(ml_results)
    bl_ds = xr.Dataset(bl_results)
    
    return ml_ds, bl_ds


def binning_2d(x,y,y_true, bins):
    """
    Binning of a third variable by two other variables. 
    """
    _,_,_,indices = binned_statistic_2d(x, y, None, 'count', 
                                    bins=bins, expand_binnumbers=True)
    x_bin_indices = indices[1,:]#-1
    y_bin_indices = indices[0,:]#-1
    for i,j in itertools.product(np.unique(y_bin_indices), np.unique(x_bin_indices)):
        where_is_item = np.where((x_bin_indices==i) & (y_bin_indices==j))[0]
        i-=1; j-=1
        skew[j,i] = np.mean(y_true[where_is_item]) if len(where_is_item) > 0 else 0.
        
    return skew 
    
    
def get_geography(X, test_dates):
    wofs_domain = pd.read_pickle('wofs_domains.pkl').astype(int)
    cent_lon = wofs_domain['cent_lon']
    cent_lat = wofs_domain['cent_lat']
    all_wofs_dates = wofs_domain['dates']
    east_west_divider = np.where(np.round(cent_lon,5) <= -90.0, 'west', 'east')
    north_south_divider = np.where(np.round(cent_lat,5) >= 37.0, 'north', 'south')
    divider = np.array([f'{a}_{b}' for a,b in zip(north_south_divider, east_west_divider)], dtype='object')
    iterator = ['Northern Great Plains', 'Mid-Atlantic', 'Southern Great Plains', 'South East']
    mapper = {'north_west' : 'Northern Great Plains', 
          'north_east' : 'Mid-Atlantic', 
          'south_west' : 'Southern Great Plains', 
          'south_east' : 'South East'
         }
    geography = np.zeros((len(X)), dtype='object')
    for a,b in itertools.product(['north', 'south'], ['east', 'west']):
        cardinality = f'{a}_{b}'
        date_subset = list(all_wofs_dates[divider==cardinality])
        geography[test_dates.astype(int).isin((date_subset))] = cardinality
    
    geography = np.array([mapper[item] for item in geography], dtype='object')
    
    return geography 

def get_init_time(X, init_times):
    # Group Initialization times into overlapping hour periods (e.g., 2100-2200, 2200-2300, )
    times_iterator = ['2000', '2030', '2100', '2130', '2200',
       '2230', '2300', '2330','0000', '0030', '0100', '0130', '0200', '0230', '0300',]
    group = []
    for i in range(len(times_iterator)-2):
        if i%2==0:
            group.append(list(times_iterator[i+d] for d in range(3)))
    
    init_times_mapper={}
    for i, g in enumerate(group):
        first_time = g[0][:2]+':'+g[0][2:]
        second_time = g[-1][:2]+':'+g[-1][2:]
        init_times_mapper[i] = f'{first_time} - {second_time}'
    
    
    init_times_mapper[-1] = '<20:00'
    
    
    df = pd.DataFrame({'init_times' : init_times.astype(str)})
    
    # The init time groups start at 20:00 so any init times 
    # before that are set as -1 
    init_time_idx = np.ones((len(X)))*-1
    for i,item in enumerate(group):
        where_is_item = df['init_times'].isin(item).values 
        init_time_idx[where_is_item] = i     

    init_time_rng = np.array([init_times_mapper[i] for i in init_time_idx], dtype='object')
    
    return init_time_rng



