import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements.
    """

    # number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def diff_of_means(data_1, data_2):
    """
    Difference in means of two arrays.
    """

    # the difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)
    
    return diff

def bootstrap_replicate_1d(data, func):
    """
    Bootstrapping - resample data.
    """
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """
    Draw bootstrap replicates.
    """

    # initialize an array of replicates
    bs_replicates = np.empty(size)

    # generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

def permutation_sample(data_1, data_2):
    """
    Generate a permutation sample from two data sets.
    """

    # concatenate the data sets: data
    data = np.concatenate((data_1, data_2))

    # permute the concatenated array
    permuted_data = np.random.permutation(data)

    # split the permuted array into two
    perm_sample_1 = permuted_data[:len(data_1)]
    perm_sample_2 = permuted_data[len(data_1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """
    Generate multiple permutation replicates.
    """

    # initialize array of replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

def pct_std_barplot(x, y, hue, data, ax=None, order=None):
    """
    Standardize by percentage using Pandas functions. 
    Plot using Seaborn.
    Function arguments are extentions of Seaborns'.
    """

    sns.barplot(x=x, y=y, hue=hue, ax=ax, order=order,
    data = (data[[x, hue]]
     .reset_index(drop=True)
     .groupby([x])[hue]
     .value_counts(normalize=True)
     .rename('Percentage').mul(100)
     .reset_index()
     .sort_values(hue)))
    plt.title("Percentage of {} by {}".format(hue, x))
    plt.ylabel("Percentage %")

def sample_distplot(replicates, empirical_value, title, xlabel, ylabel, null_value=None):
    """
    A detailed histogram of sampling distribution.
    """

    ptile_low = np.percentile(replicates, 2.5) # 2.5th percentile
    ptile_high = np.percentile(replicates, 97.5) # 97.5th percentile
    fig = plt.figure(figsize = (8,5))
    plt.hist(replicates)
    plt.axvline(empirical_value, color='red', label='Empirical Value')
    if null_value is not None:
        plt.axvline(null_value, color='orange', label='Null Hypothesis Value')
    plt.axvline(ptile_low, color='yellow', label='2.5th Percentile')
    plt.axvline(ptile_high, color='yellow', label='97.5th Percentile')
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='center left', 
               bbox_to_anchor=(1, 0.5), 
               fancybox=True, 
               shadow=True,
               frameon=True,
               fontsize=12)
    plt.show()