#============================
# VERIFICATION METRICS 
#============================

import numpy as np

from scipy import interpolate
from mlxtend.evaluate import permutation_test
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score, precision_recall_curve

from functools import partial
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target
from mlxtend.evaluate import permutation_test


def stat_testing(new_score, baseline_score):
    """
    Compute a p-value between two sets using permutation testing 
    to determined statistical significance. In this case,
    assess whether the ML performance is greater than the baseline.
    """
    p_value = permutation_test(new_score,
                              baseline_score,
                             'x_mean != y_mean',
                              method='approximate',
                               num_rounds=1000,
                               seed=0)
    return p_value

def modified_precision(precision, known_skew, new_skew): 
    """
    Modify the success ratio according to equation (3) from 
    Lampert and Gancarski (2014). 
    
                       pi'
    SR' = -------------------------------
         pi' + (1-pi')(pi/1-pi)*((1/SR)-1)
    
    where pi' is the skew to be transformed into (the known skew in most cases)
    and pi is the existing skew of the dataset. 
    
    pi' : known_skew
    pi  : new_skew
    
    """
    precision[precision<1e-5] = 1e-5
    term1 = new_skew / (1.0-new_skew)
    term2 = ((1/precision) - 1.0)

    denom = known_skew + ((1-known_skew)*term1*term2)
    
    return known_skew / denom 
    
def calc_sr_min(skew):
    pod = np.linspace(0,1,100)
    sr_min = (skew*pod) / (1-skew+(skew*pod))
    return sr_min 

def _binary_uninterpolated_average_precision(
            y_true, y_score, known_skew, new_skew, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        if known_skew is not None:
            precision = modified_precision(precision, known_skew, new_skew)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def min_aupdc(y_true, pos_label, average, sample_weight=None, known_skew=None, new_skew=None):
    """
    Compute the minimum possible area under the performance 
    diagram curve. Essentially, a vote of NO for all predictions. 
    """
    min_score = np.zeros((len(y_true)))
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    ap_min = _average_binary_score(average_precision, y_true, min_score,
                                 average, sample_weight=sample_weight)

    return ap_min
        
def norm_aupdc(y_true, y_score, known_skew=None, *, average="macro", pos_label=1,
                            sample_weight=None, min_method='random'):
    """
    Compute the normalized modified average precision. Normalization removes 
    the no-skill region either based on skew or random classifier performance. 
    Modification alters success ratio to be consistent with a known skew. 
  
    Parameters:
    -------------------
        y_true, array of (n_samples,)
            Binary, truth labels (0,1)
        y_score, array of (n_samples,)
            Model predictions (either determinstic or probabilistic)
        known_skew, float between 0 and 1 
            Known or reference skew (# of 1 / n_samples) for 
            computing the modified success ratio.
        min_method, 'skew' or 'random'
            If 'skew', then the normalization is based on the minimum AUPDC 
            formula presented in Boyd et al. (2012).
            
            If 'random', then the normalization is based on the 
            minimum AUPDC for a random classifier, which is equal 
            to the known skew. 
    
    
    Boyd, 2012: Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation, ArXiv
    """
    new_skew = np.mean(y_true)
    if known_skew is None:
        known_skew = new_skew
    
    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    
    ap = _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
    
    if min_method == 'random':
        ap_min = known_skew 
    elif min_method == 'skew':
        ap_min = min_aupdc(y_true, 
                       pos_label, 
                       average,
                       sample_weight=sample_weight,
                       known_skew=known_skew, 
                       new_skew=new_skew)
    
    ##print(f'{ap=}', f'{ap_min=}')
    
    naupdc = (ap - ap_min) / (1.0 - ap_min)

    return naupdc

def brier_skill_score(y_values, forecast_probabilities, **kwargs):
    """Computes the brier skill score"""
    climo = np.mean((y_values - np.mean(y_values)) ** 2)
    return 1.0 - brier_score_loss(y_values, forecast_probabilities) / climo


def bss_reliability(targets, predictions):
    """
    Reliability component of BSS. Weighted MSE of the mean forecast probabilities
    and the conditional event frequencies. 
    """
    mean_fcst_probs, event_frequency, indices = reliability_curve(targets, predictions, n_bins=10)
    # Add a zero for the origin (0,0) added to the mean_fcst_probs and event_frequency
    counts = [1e-5]
    for i in indices:
        if i is np.nan:
            counts.append(1e-5)
        else:
            counts.append(len(i[0]))

    mean_fcst_probs[np.isnan(mean_fcst_probs)] = 1e-5
    event_frequency[np.isnan(event_frequency)] = 1e-5

    diff = (mean_fcst_probs-event_frequency)**2
    return np.average(diff, weights=counts)

def reliability_curve(targets, predictions, n_bins=10):
        """
        Generate a reliability (calibration) curve. 
        Bins can be empty for both the mean forecast probabilities 
        and event frequencies and will be replaced with nan values. 
        Unlike the scikit-learn method, this will make sure the output
        shape is consistent with the requested bin count. The output shape
        is (n_bins+1,) as I artifically insert the origin (0,0) so the plot
        looks correct. 
        """
        bin_edges = np.linspace(0,1, n_bins+1)
        bin_indices = np.clip(
                np.digitize(predictions, bin_edges, right=True) - 1, 0, None
                )

        indices = [np.where(bin_indices==i+1)
               if len(np.where(bin_indices==i+1)[0]) > 0 else np.nan for i in range(n_bins) ]

        mean_fcst_probs = [np.nan if i is np.nan else np.mean(predictions[i]) for i in indices]
        event_frequency = [np.nan if i is np.nan else np.sum(targets[i]) / len(i[0]) for i in indices]

        # Adding the origin to the data
        mean_fcst_probs.insert(0,0)
        event_frequency.insert(0,0)
        
        return np.array(mean_fcst_probs), np.array(event_frequency), indices
