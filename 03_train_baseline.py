import sys
import numpy as np
from config import conf
import getopt
from sklearn.cross_validation import LabelKFold as LKF
from sklearn.cross_validation import StratifiedKFold as SKF
from sklearn.metrics import f1_score, accuracy_score
import pandas as pns

def load_data(ws, t):
	_, y_file, id_file = conf.get_merged_feature_files(ws)
	y_ws = np.genfromtxt(y_file, delimiter=',', skip_header=1).astype(int)[:,t]
	ids_ws = np.genfromtxt(id_file, delimiter=',', skip_header=1).astype(int)[:,0]
	return y_ws, ids_ws

def get_baseline_f1_score(t):
	"""
	train a baseline classifier and return the F1 score it achieves
	"""
	outer_cv = SKF(participant_scores, conf.n_outer_folds, shuffle=True)

	preds = np.zeros((conf.n_participants), dtype=int)
	truth = np.zeros((conf.n_participants), dtype=int)

	for outer_i, (outer_train_participants, outer_test_participants) in enumerate(outer_cv):
		inner_performance = np.zeros((conf.n_inner_folds, len(conf.all_window_sizes)))

		for ws_i in xrange(0, len(conf.all_window_sizes)):
			ws = conf.all_window_sizes[ws_i]

			# load data for this window size
			y_ws, ids_ws = load_data(ws, t)

			# cut out the outer train samples
			outer_train_samples = np.array([p in outer_train_participants for p in ids_ws])
			outer_train_y = y_ws[outer_train_samples]
			outer_train_y_ids = ids_ws[outer_train_samples]

			# build inner cross validation such that all samples of one person are either in training or testing
			inner_cv = LKF(outer_train_y_ids, n_folds=conf.n_inner_folds)
			for inner_i, (inner_train_indices, inner_test_indices) in enumerate(inner_cv):
				# create inner train and test samples. Note: both are taken from outer train samples!
				inner_y_train = outer_train_y[inner_train_indices]
				unique_inner_test_ids = np.unique(outer_train_y_ids[inner_test_indices])

				# predict the most frequent class from the training set
				hist,_ = np.histogram(inner_y_train, bins=[0.5,1.5,2.5,3.5])
				guess = np.argmax(hist) + 1
				innerpreds = np.full(len(unique_inner_test_ids), guess, dtype=int)
				innertruth = participant_scores[unique_inner_test_ids]

				inner_performance[inner_i, ws_i] = accuracy_score(np.array(innertruth), np.array(innerpreds))

		# evaluate classifier on outer cv using the best window size from inner cv
		chosen_ws_i = np.argmax(np.mean(inner_performance, axis=0))
		chosen_ws = conf.all_window_sizes[chosen_ws_i]
		y, ids = load_data(chosen_ws, t)

		outer_train_samples = np.array([p in outer_train_participants for p in ids])
		outer_test_samples = np.array([p in outer_test_participants for p in ids])

		if outer_train_samples.size > 0 and outer_test_samples.size > 0:
			y_train = y[outer_train_samples]

			# guess the most frequent class
			hist,_ = np.histogram(y_train, bins=[0.5, 1.5, 2.5, 3.5])
			guess = np.argmax(hist) + 1

			for testp in outer_test_participants:
				if testp in ids[outer_test_samples]:
					preds[testp] = guess
					truth[testp] = participant_scores[testp]
				else:
					# participant does not occour in outer test set, e.g. because their time in the shop was too short
					preds[testp] = -1
					truth[testp] = -1
					print 'not enough samples for participant', testp
			#print 'preds collected'
		else:
			for testp in outer_test_participants:
				preds[testp] = np.array([])
				truth[testp] = -1

	f1 = f1_score(truth, preds, average='macro')
	return f1

# If the program is run directly:
if __name__ == "__main__":
	df = []
	for trait in xrange(0, conf.n_traits):
		participant_scores = np.loadtxt(conf.binned_personality_file, delimiter=',', skiprows=1, usecols=(trait+1,))
		print conf.medium_traitlabels[trait]
		for si in xrange(0,conf.max_n_iter):
			f1 = get_baseline_f1_score(trait)
			print '\t'+str(si)+':', f1
			df.append([f1, conf.medium_traitlabels[trait], si])
	df_pns = pns.DataFrame(data=df, columns=['F1', 'trait', 'iteration'])
	df_pns.to_csv(conf.result_folder + '/most_frequ_class_baseline.csv')
	print conf.result_folder + '/most_frequ_class_baseline.csv written.'
