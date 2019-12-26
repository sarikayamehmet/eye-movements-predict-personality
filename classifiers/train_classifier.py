import sys
import numpy as np
from config import conf 
import os
import getopt
import threading
from sklearn.cross_validation import LabelKFold as LKF
from sklearn.cross_validation import StratifiedKFold as SKF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


def predict_all():
	# add threads to a list, and wait for all of them in the end
	threads = []

	for trait in trait_list:
		for si in xrange(low_repetitions, num_repetitions):
			fname = conf.get_result_filename(annotation_value, trait, shuffle_labels, si, add_suffix=True)
			if not os.path.exists(fname):
				thread = threading.Thread(target=save_predictions, args=(trait, conf.get_result_filename(annotation_value, trait, shuffle_labels, si), si))
				sys.stdout.flush()
				thread.start()
				threads.append(thread)
			else:
				print "existing solution:", fname

	for thread in threads:
		thread.join()
		print 'waiting to join'

def load_data(ws, annotation_value, t, chosen_features = None):
	x_file, y_file, id_file = conf.get_merged_feature_files(ws)
	if annotation_value == conf.annotation_all:
		x_ws = np.genfromtxt(x_file, delimiter=',', skip_header=1)
		y_ws = np.genfromtxt(y_file, delimiter=',', skip_header=1).astype(int)[:,t]
		ids_ws = np.genfromtxt(id_file, delimiter=',', skip_header=1).astype(int)[:,0]
	elif annotation_value == conf.annotation_shop:
		x_ws = np.genfromtxt(x_file, delimiter=',', skip_header=1)
		y_ws = np.genfromtxt(y_file, delimiter=',', skip_header=1).astype(int)[:,t]
		ids_ws = np.genfromtxt(id_file, delimiter=',', skip_header=1).astype(int)

		x_ws = x_ws[ids_ws[:,1] == conf.time_window_annotation_shop,:]
		y_ws = y_ws[ids_ws[:,1] == conf.time_window_annotation_shop]
		ids_ws = ids_ws[ids_ws[:,1] == conf.time_window_annotation_shop,0]
	elif annotation_value == conf.annotation_ways:
		x_ws = np.genfromtxt(x_file, delimiter=',', skip_header=1)
		y_ws = np.genfromtxt(y_file, delimiter=',', skip_header=1).astype(int)[:,t]
		ids_ws = np.genfromtxt(id_file, delimiter=',', skip_header=1).astype(int)

		x_ws = x_ws[(ids_ws[:,1] == conf.time_window_annotation_wayI) | (ids_ws[:,1] == conf.time_window_annotation_wayII),:]
		y_ws = y_ws[(ids_ws[:,1] == conf.time_window_annotation_wayI) | (ids_ws[:,1] == conf.time_window_annotation_wayII)]
		ids_ws = ids_ws[(ids_ws[:,1] == conf.time_window_annotation_wayI) | (ids_ws[:,1] == conf.time_window_annotation_wayII),0]
	else:
		print 'unknown annotation value', annotation_value
		print 'should be 0 (all data), 1 (way) or 2 (shop).'
		sys.exit(1)
	if chosen_features is not None:
		x_ws = x_ws[:,chosen_features]
	return x_ws, y_ws, ids_ws


def save_predictions(t, filename, rs):
	"""
	train a classifier and write results to file
	"""
	# create RandomForest classifier with parameters given in _conf.py
	clf = RandomForestClassifier(random_state=rs, verbose=verbosity, class_weight='balanced',
	                             n_estimators=conf.n_estimators, n_jobs=conf.max_n_jobs, max_features=conf.tree_max_features,
	                             max_depth=conf.tree_max_depth)

	# create StandardScaler that will be used to scale each feature
	# such that it has mean 0 and std 1 on the trianing set
	scaler = StandardScaler(with_std=True, with_mean=True)

	# use ground truth to create folds for outer cross validation in a stratified way, i.e. such that
	# each label occurs equally often
	participant_scores = np.loadtxt(conf.binned_personality_file, delimiter=',', skiprows=1, usecols=(t+1,))
	outer_cv = SKF(participant_scores, conf.n_outer_folds, shuffle=True)

	# initialise arrays to save information
	feat_imp = np.zeros((len(outer_cv), conf.max_n_feat))  # feature importance
	preds = np.zeros((conf.n_participants), dtype=int)  # predictions on participant level
	detailed_preds = np.zeros((conf.n_participants), dtype=object)  # predictions on time window level, array of lists
	chosen_ws_is = np.zeros((conf.n_participants), dtype=int)  # indices of window sizes chosen in the inner cross validation

	for outer_i, (outer_train_participants, outer_test_participants) in enumerate(outer_cv):
		print
		print str(outer_i + 1) + '/' + str(conf.n_outer_folds)

		# find best window size in inner cv, and discard unimportant features
		inner_performance = np.zeros((conf.n_inner_folds, len(all_window_sizes)))
		inner_feat_importances = np.zeros((conf.max_n_feat, len(all_window_sizes)))

		for ws_i in xrange(0, len(all_window_sizes)):
			ws = all_window_sizes[ws_i]
			print '\t', 'ws ' + str(ws_i + 1) + '/' + str(len(all_window_sizes))

			# load data for this window size
			x_ws, y_ws, ids_ws = load_data(ws, annotation_value, t)
			if shuffle_labels:
				np.random.seed(316588 + 111 * t + rs)
				perm = np.random.permutation(len(y_ws))
				y_ws = y_ws[perm]
				ids_ws = ids_ws[perm]

			# cut out the outer train samples
			outer_train_samples = np.array([p in outer_train_participants for p in ids_ws])
			outer_train_x = x_ws[outer_train_samples, :]
			outer_train_y = y_ws[outer_train_samples]
			outer_train_y_ids = ids_ws[outer_train_samples]

			# build inner cross validation such that all samples of one person are either in training or testing
			inner_cv = LKF(outer_train_y_ids, n_folds=conf.n_inner_folds)
			for inner_i, (inner_train_indices, inner_test_indices) in enumerate(inner_cv):
				# create inner train and test samples. Note: both are taken from outer train samples!
				inner_x_train = outer_train_x[inner_train_indices, :]
				inner_y_train = outer_train_y[inner_train_indices]

				inner_x_test = outer_train_x[inner_test_indices, :]
				inner_y_test = outer_train_y[inner_test_indices]

				# fit scaler on train set and scale both train and test set with the result
				scaler.fit(inner_x_train)
				inner_x_train = scaler.transform(inner_x_train)
				inner_x_test = scaler.transform(inner_x_test)

				# fit Random Forest
				clf.fit(inner_x_train, inner_y_train)

				# save predictions and feature importance
				inner_pred = clf.predict(inner_x_test)
				inner_feat_importances[:, ws_i] += clf.feature_importances_

				# compute and save performance in terms of accuracy
				innerpreds = []
				innertruth = []
				inner_test_ids = outer_train_y_ids[inner_test_indices]
				for testp in np.unique(inner_test_ids):
					(values, counts) = np.unique(inner_pred[inner_test_ids == testp], return_counts=True)
					ind = np.argmax(counts)
					innerpreds.append(values[ind])
					innertruth.append(inner_y_test[inner_test_ids == testp][0])
				inner_performance[inner_i, ws_i] = accuracy_score(np.array(innertruth), np.array(innerpreds))
				print '                ACC: ', '%.2f' % (inner_performance[inner_i, ws_i] * 100)

		# evaluate classifier on outer cv using the best window size from inner cv, and the most informative features
		chosen_ws_i = np.argmax(np.mean(inner_performance, axis=0))
		chosen_ws = all_window_sizes[chosen_ws_i]
		chosen_features = (inner_feat_importances[:,chosen_ws_i]/float(conf.n_inner_folds)) > 0.005

		# reload all data
		x, y, ids = load_data(chosen_ws, annotation_value, t, chosen_features=chosen_features)
		if shuffle_labels:
			np.random.seed(316588 + 111 * t + rs + 435786)
			perm = np.random.permutation(len(y))
			y = y[perm]
			ids = ids[perm]

		outer_train_samples = np.array([p in outer_train_participants for p in ids])
		outer_test_samples = np.array([p in outer_test_participants for p in ids])

		if outer_train_samples.size > 0 and outer_test_samples.size > 0:
			x_train = x[outer_train_samples, :]
			y_train = y[outer_train_samples]

			x_test = x[outer_test_samples, :]
			y_test = y[outer_test_samples]

			# scaling
			scaler.fit(x_train)
			x_train = scaler.transform(x_train)
			x_test = scaler.transform(x_test)

			# fit Random Forest
			clf.fit(x_train, y_train)
			pred = clf.predict(x_test)

			for testp in outer_test_participants:
				chosen_ws_is[testp] = chosen_ws_i
				if testp in ids[outer_test_samples]:
					# majority voting over all samples that belong to participant testp
					(values, counts) = np.unique(pred[ids[outer_test_samples] == testp], return_counts=True)
					ind = np.argmax(counts)
					preds[testp] = values[ind]
					detailed_preds[testp] = list(pred[ids[outer_test_samples] == testp])
				else:
					# participant does not occour in outer test set, e.g. because their time in the shop was too short
					preds[testp] = -1
					detailed_preds[testp] = []

			# save the resulting feature importance
			feat_imp[outer_i, chosen_features] = clf.feature_importances_

		else:
			for testp in outer_test_participants:
				chosen_ws_is[testp] = -1
				preds[testp] = np.array([])
				truth[testp] = -1
			feat_imp[outer_i, chosen_features] = -1

	# compute resulting F1 score and save to file
	nonzero_preds = preds[preds>0]
	nonzero_truth = participant_scores[preds>0]
	f1 = f1_score(nonzero_truth, nonzero_preds, average='macro')
	np.savez(filename, f1=f1, predictions=preds, chosen_window_indices=chosen_ws_is,
				feature_importances=feat_imp, detailed_predictions=detailed_preds)
	print f1, 'written', filename

# If the program is run directly:
if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:], "t:m:l:s:a:", [])
	except getopt.GetoptError:
		print 'valid arguments:'
		print '-t 	trait index'
		print '-s   1 to perform label permutation test, do not pass s or use -s 0 otherwise'
		print '-l   lowest number of repetitions'
		print '-m   max number of repetitions'
		print '-a   using partial data only: 0 (all data), 1 (way data), 2(shop data)'
		sys.exit(2)


	low_repetitions = 0
	num_repetitions = conf.max_n_iter
	verbosity = 0
	shuffle_labels = False
	annotation_value = conf.annotation_all
	trait_list = xrange(0, conf.n_traits)

	for opt, arg in opts:
		if opt == '-t':
			t = int(arg)
			assert t in trait_list
			trait_list = [t]
		elif opt == '-a':
			annotation_value = int(arg)
			assert annotation_value in conf.annotation_values
		elif opt == '-s':
			shuffle_labels = bool(int(arg))
		elif opt == '-m':
			num_repetitions = int(arg)
		elif opt == '-l':
			low_repetitions = int(arg)
		else:
			print 'valid arguments:'
			print '-t 	trait index'
			print '-s   1 to perform label permutation test, do not pass s or use -s 0 otherwise'
			print '-l   lowest number of repetitions'
			print '-m   max number of repetitions'
			print '-a   using partial data only: 0 (all data), 1 (way data), 2(shop data)'
			sys.exit(2)

	result_folder = conf.get_result_folder(annotation_value)
	if not os.path.exists(result_folder):
		os.makedirs(result_folder)

	# restrict window sizes in case shop data should be used
	if annotation_value == conf.annotation_shop:
		all_window_sizes = conf.all_shop_window_sizes
	else:
		all_window_sizes = conf.all_window_sizes

	predict_all()
