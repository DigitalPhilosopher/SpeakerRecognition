import os
import math
from typing import List, Dict, Optional, Union

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import roc_curve

def plot_similarity_lists_bar(list_list: List[List[float]], name_list: List[str] = None, do_plot: bool = True,
                              bins: int = 25, save_plot_path=None, xlabel:str = 'similarity score', normalize_scores:bool = False) -> [List[List[int]], List[float]]:
    """
    This function is used to plot the score distribution given by the similarity lists contained in list_list
    Args:
        list_list: A list of multiple score lists
        name_list: A list of names corresponding to a list in list_list (used for visualisation)
        do_plot: Whether to plot the distribution or not
        bins: Number of bins in the plot
        xlabel: label of the x-axis

    Returns:
        frequency_list_list: A list containing lists of the the frequencys on how often the score of the corresponding list in list_list appeared in a bin
        score_value_list: The list with length = bins of scores on which the scores are mapped to.
    """
    # list_data_dict = {
    #     'score_list': list_list,
    #     'name_list': name_list
    # }
    # np.save(os.path.join(os.path.dirname(save_plot_path), "plot_data.npy"), list_data_dict)

    def filter_out_none(list_list: List[List[float]]) -> List[List[float]]:
        """ Filter out None values form the Lists"""
        return [[item for item in sublist if item is not None] for sublist in list_list]
    list_list = filter_out_none(list_list)
    num_bins = 250



    min_l = math.inf
    max_l = -1 * math.inf
    for list_scores in list_list:
        if float(min(list_scores)) < min_l:
            min_l = float(min(list_scores))
        if float(max(list_scores)) > max_l:
            max_l = float(max(list_scores))
    if normalize_scores:
        max_score = max_l
        min_score = min_l
        min_l = 0
        max_l = 1

    max_y = 0
    color_list_negative = ["red", "purple"]
    color_list_positive = ["green", "blue", "cyan"]
    color_list_neutral = ["orange", "black", "yellow"]

    for i, list_scores in enumerate(list_list):
        if normalize_scores:
            list_scores = [(x-min_score)/(max_score-min_score) for x in list_scores]
        if "genuine" in name_list[i].lower():
            color = color_list_positive[0]
            color_list_positive.pop(0)
        elif "deepfake" in name_list[i].lower() or "attack" in name_list[i].lower() or "pa" in name_list[i].lower() or "far" in name_list[i].lower() or "false" in name_list[i].lower():
            color = color_list_negative[0]
            color_list_negative.pop(0)
        else:
            color = color_list_neutral[0]
            color_list_neutral.pop(0)

        counts, bins = np.histogram(list_scores, bins=num_bins, range=(min_l, max_l))
        normalize_factor = len(list_scores)
        counts = [(count / normalize_factor) for count in counts]

        if max(counts) > max_y:
            max_y = max(counts)
        pyplot.hist(bins[:-1], bins, weights=counts, color=color, alpha=0.5, label=name_list[i])
    pyplot.ylim(0, max_y)
    # plt.yscale('log')
    pyplot.legend(loc='upper right')
    pyplot.xlabel(xlabel)
    if do_plot:
        pyplot.show()
    if save_plot_path is not None:
        pyplot.savefig(save_plot_path)
    pyplot.clf()

    return None, None


def calc_eer(
        score_list_genuine:List[float],
        score_list_imposter:List[float]
) -> float:
    """
    Calculates the eer based on genuine and imposter lists
    Args:
        threshold: the threshold of the asv model to be used
        target: whether or not the similarity should be calculated to the target speaker (True) or to the soucre speaker (False)
    Returns:
        the average matching rate

    """
    y = [1] * len(score_list_imposter) + [0] * len(score_list_genuine)
    y_pred = score_list_imposter + score_list_genuine

    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = (fpr[np.nanargmin(np.absolute((fnr - fpr)))] + fnr[np.nanargmin(np.absolute((fnr - fpr)))]) / 2
    return EER