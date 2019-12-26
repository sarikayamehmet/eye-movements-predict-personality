import numpy as np
from config import conf
import os, sys
from config import names as gs
import pandas as pd

truth = np.genfromtxt(conf.binned_personality_file, skip_header=1, usecols=xrange(1, conf.n_traits+1), delimiter=',')

# all comparisons to perform. Each has
#     a name,
#     two annotation values that determine if classifiers trained on all data or on specific subsets only will be examined;
#     names for both tasks to compare
comparisons = dict({'split halves': [conf.annotation_all, conf.annotation_all, 'first half', 'second half'],
                    'two ways': [conf.annotation_ways, conf.annotation_ways, 'way there', 'way back'],
                    'way vs shop in general classifier': [conf.annotation_all, conf.annotation_all, 'both ways' ,'shop'],
                    'way vs shop in specialised classifier': [conf.annotation_ways, conf.annotation_shop, 'both ways', 'shop'],
                    'way in specialised classifier vs way in general classifier': [conf.annotation_ways, conf.annotation_all, 'both ways', 'both ways'],
                    'shop in specialised classifier vs shop in general classifier': [conf.annotation_shop, conf.annotation_all, 'shop', 'shop']
                    })

def get_majority_vote(predictions):
    if len(predictions) == 0:
        return -1
    (values, counts) = np.unique(predictions, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def get_average_correlation(predA, predB, m_iter):
    """
    :param predA: predictions for task A, n_participants x m_iter
    :param predB: predictions for task B, n_participants x m_iter
    :return:
    """
    correlations = []
    for si in xrange(0, m_iter):
        if predB.ndim == 1:
            if np.sum(predA[:,si]) > 0:
                A = predA[:,si]
                B = predB
                consider = (A>0)
                A = A[consider]
                B = B[consider]
            else:
                continue
        else:
            if np.sum(predA[:,si]) > 0 and (np.sum(predB[:,si]) > 0):
                A = predA[:,si]
                B = predB[:,si]
                consider = (A>0) & (B>0)
                A = A[consider]
                B = B[consider]
            else:
                continue

        correlation = np.corrcoef(np.array([A, B]))[0][1]
        correlations.append(correlation)

    avg = np.tanh(np.mean(np.arctanh(np.array(correlations))))
    return avg


if __name__ == "__main__":
    # check if the output target folder already exists and create if not
    if not os.path.exists(conf.figure_folder):
        os.mkdir(conf.figure_folder)

    # collect masks for each participant, annotation (all data, shop, way), window size and subset in question (e.g. first half, or way to the shop)
    # each mask is True for samples of a particular participant and subset; False for all others
    window_masks = []
    for wsi in xrange(0, len(conf.all_window_sizes)):
        x_file, y_file, id_file = conf.get_merged_feature_files(conf.all_window_sizes[wsi])

        for annotation_value in conf.annotation_values:
            ids_ws = np.genfromtxt(id_file, delimiter=',', skip_header=1).astype(int)

            if annotation_value == conf.annotation_shop:
                ids_ws = ids_ws[ids_ws[:, 1] == conf.time_window_annotation_shop, :]
            elif annotation_value == conf.annotation_ways:
                ids_ws = ids_ws[(ids_ws[:, 1] == conf.time_window_annotation_wayI) | (ids_ws[:, 1] == conf.time_window_annotation_wayII), :]

            for p in xrange(0, conf.n_participants):
                ids_ws_p = ids_ws[(ids_ws[:, 0] == p), :]

                window_masks.append([annotation_value, p, wsi, 'first half', ids_ws_p[:, 2] == conf.time_window_annotation_halfI])
                window_masks.append([annotation_value, p, wsi, 'second half', ids_ws_p[:, 2] == conf.time_window_annotation_halfII])

                window_masks.append([annotation_value, p, wsi, 'way there', ids_ws_p[:, 1] == conf.time_window_annotation_wayI])
                window_masks.append([annotation_value, p, wsi, 'way back', ids_ws_p[:, 1] == conf.time_window_annotation_wayII])

                window_masks.append([annotation_value, p, wsi, 'shop', ids_ws_p[:, 1] == conf.time_window_annotation_shop])
                window_masks.append([annotation_value, p, wsi, 'both ways', np.logical_or(ids_ws_p[:, 1] == conf.time_window_annotation_wayI,ids_ws_p[:, 1] == conf.time_window_annotation_wayII)])

    window_masks_df = pd.DataFrame(window_masks, columns=['annotation', 'participant', 'window size index', 'subtask', 'mask'])

    # collect predictions for each participant and each setting that is interesting for one of the comparisons
    # Results are directly written into figures/table1-5.csv
    with open(conf.figure_folder + '/table1-5.csv', 'w') as f:
        f.write('comparison')
        for trait in xrange(0, conf.n_traits):
            f.write(',' + conf.medium_traitlabels[trait])
        f.write('\n')

        for comp_title, (annotation_value_I, annotation_value_II, subtaskI, subtaskII) in comparisons.items():
            f.write(comp_title)
            result_filename = conf.result_folder + '/predictions_' + comp_title.replace(' ','_') + '.npz'
            if not os.path.exists(result_filename):
                print 'computing data for', comp_title
                print 'Note taht this might take a while - if the script is run again, intermediate results will be available and speed up all computations.'

                predictions_I = np.zeros((conf.n_participants, conf.n_traits, conf.max_n_iter), dtype=int)
                predictions_II = np.zeros((conf.n_participants, conf.n_traits, conf.max_n_iter), dtype=int)

                for trait in xrange(0, conf.n_traits):
                    for si in xrange(0, conf.max_n_iter):
                        filenameI = conf.get_result_filename(annotation_value_I, trait, False, si, add_suffix=True)
                        filenameII = conf.get_result_filename(annotation_value_II, trait, False, si, add_suffix=True)

                        if os.path.exists(filenameI) and os.path.exists(filenameII):
                            dataI = np.load(filenameI)
                            detailed_predictions_I = dataI['detailed_predictions']
                            chosen_window_indices_I = dataI['chosen_window_indices']

                            dataII = np.load(filenameII)
                            detailed_predictions_II = dataII['detailed_predictions']
                            chosen_window_indices_II = dataII['chosen_window_indices']

                            for p, window_index_I, window_index_II, local_detailed_preds_I, local_detailed_preds_II in zip(xrange(0, conf.n_participants), chosen_window_indices_I, chosen_window_indices_II, detailed_predictions_I, detailed_predictions_II):
                                maskI = window_masks_df[(window_masks_df.annotation == annotation_value_I) &
                                                        (window_masks_df.participant == p) &
                                                        (window_masks_df['window size index'] == window_index_I) &
                                                        (window_masks_df.subtask == subtaskI)
                                                        ].as_matrix(columns=['mask'])[0][0]
                                maskII = window_masks_df[(window_masks_df.annotation == annotation_value_II) &
                                                        (window_masks_df.participant == p) &
                                                        (window_masks_df['window size index'] == window_index_II) &
                                                        (window_masks_df.subtask == subtaskII)
                                                        ].as_matrix(columns=['mask'])[0][0]

                                predictions_I[p, trait, si] = get_majority_vote(np.array(local_detailed_preds_I)[maskI])
                                predictions_II[p, trait, si] = get_majority_vote(np.array(local_detailed_preds_II)[maskII])
                        else:
                            print 'did not find', filenameI, 'or', filenameII
                            sys.exit(1)
                np.savez(result_filename, predictions_I=predictions_I, predictions_II=predictions_II)
            else:
                data = np.load(result_filename)
                predictions_I = data['predictions_I']
                predictions_II = data['predictions_II']

            # predictions_I are predictions from one context, predictions_II is the other context
            # compute their average correlation and write it to file
            for t in xrange(0, conf.n_traits):
                corrI = get_average_correlation(predictions_I[:, t, :], predictions_II[:, t, :], 100)
                f.write(','+'%.2f'%corrI)
            f.write('\n')
