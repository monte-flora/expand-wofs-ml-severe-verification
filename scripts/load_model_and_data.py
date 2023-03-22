import pandas as pd 
from joblib import load
from glob import glob
from os.path import join 
import sys
sys.path.append('/home/monte.flora/python_packages/master/ml_workflow')


baseline_names = {
                'tornado': 'Baseline__tornado', 
                 'severe_hail' : 'Baseline__severe_hail',
                'severe_wind' : 'Baseline__severe_wind',
}

known_skew = {'first_hour': {'severe_hail': 0.0391880873707248,
  'severe_wind': 0.027375770765324627,
  'tornado': 0.012250931885049705},
 'second_hour': {'severe_hail': 0.03567197119293762,
  'severe_wind': 0.02379619369012823,
  'tornado': 0.009605216107852312}}


def loader(lead_time, target, return_model=False):

    name = 'LogisticRegression'
    path = '/work/mflora/ML_DATA/test_data'
    test_df = pd.read_feather(join(path, f'test_ml_dataset_{lead_time}_{target}.feather'))
    init_times = test_df['Run Time']
    test_dates = test_df['Run Date']
    fti = test_df['FCST_TIME_IDX'].astype(int)
    
    # Just to keep the filename on a single line. 
    model_fname = glob(join('/work/mflora/ML_DATA/MODEL_SAVES', f'LogisticRegression_{lead_time}_{target}*'))

    org_model_fname = [f for f in model_fname if 'manual' not in f][0]
    new_model_fname = [f for f in model_fname if 'manual' in f][0]
    
    org_data = load(org_model_fname)
    original_features = org_data['features']
    model = org_data['model']

    new_data = load(new_model_fname)
    new_original_features = new_data['features']
    new_model = new_data['model']
    
    
    X = test_df[original_features]
    X_new = test_df[new_original_features]
    y = test_df[f'matched_to_{target}_0km'].values
    
    # Compute the ML predictions
    ml_predictions = model.predict_proba(X)[:,1]
    new_ml_predictions = new_model.predict_proba(X_new)[:,1]
    
    bl_predictions = test_df[baseline_names[target]].values
    
    if return_model:
        return (name, model), [ml_predictions, new_ml_predictions, bl_predictions], X, y, init_times, test_dates, fti
        
    
    return [ml_predictions, new_ml_predictions, bl_predictions], X, y, init_times, test_dates, fti