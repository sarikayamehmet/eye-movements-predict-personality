import numpy as np
import os
from config import conf
from featureExtraction import gaze_analysis as ga
import threading
import getopt
import sys
from config import names as gs

def compute_sliding_window_features(participant, ws, gazeAnalysis_instance):
	"""
	calls the gazeAnalysis instance it was given, calls it to get features and saves those to file
	"""
	window_features, window_times = gazeAnalysis_instance.get_window_features(ws, conf.get_step_size(ws))
	np.save(conf.get_window_features_file(participant, ws), window_features)
	np.save(conf.get_window_times_file(participant, ws), window_times)

if __name__ == "__main__":
	for p in xrange(0,conf.n_participants):
		threads = []  # one thread per time window will be used and collected in this list

		# create data folder, plus one subfolder for participant p
		if not os.path.exists(conf.get_feature_folder(p)):
			os.makedirs(conf.get_feature_folder(p))

		# make sure all relevant raw data files exist in the right folder
		gaze_file = conf.get_data_folder(p) + '/gaze_positions.csv'
		pupil_diameter_file = conf.get_data_folder(p) + '/pupil_diameter.csv'
		events_file = conf.get_data_folder(p) + '/events.csv'
		assert os.path.exists(gaze_file) and os.path.exists(pupil_diameter_file) and os.path.exists(events_file)

		# load relevant data
		gaze = np.genfromtxt(gaze_file, delimiter=',', skip_header=1)
		pupil_diameter = np.genfromtxt(pupil_diameter_file, delimiter=',', skip_header=1)
		events = np.genfromtxt(events_file, delimiter=',', skip_header=1, dtype=str)

		# create instance of gazeAnalysis class that will be used for feature extraction
		# this already does some initial computation that will be useful for all window sizes:
		extractor = ga.gazeAnalysis(gaze, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
													conf.saccade_min_velocity, conf.max_saccade_duration,
													pupil_diameter=pupil_diameter, event_strings=events)

		# compute sliding window features by creating one thread per window size
		for window_size in conf.all_window_sizes:
			if not os.path.exists(conf.get_window_features_file(p, window_size)):
				thread = threading.Thread(target=compute_sliding_window_features, args=(p, window_size, extractor))
				thread.start()
				threads.append(thread)

		for t in threads:
			t.join()

		print ('finished all features for participant', p)

	# Merge the features from all participants into three files per window_size:
	# merged_features includes all features
	# merged_traits contains the ground truth personality score ranges
	# merged_ids contains the participant number and context (way, shop, half of the recording)

	# load ground truth from info folder:
	binned_personality = np.genfromtxt(conf.binned_personality_file, delimiter=',', skip_header=1)
	trait_labels = np.loadtxt(conf.binned_personality_file, delimiter=',', dtype=str)[0,:]
	annotation = np.genfromtxt(conf.annotation_path, delimiter=',', skip_header=1)

	for window_size in conf.all_window_sizes:
		print ('merging window size', window_size)

		windowfeats_subtask_all = []
		windowfeats_subtask_ids = []
		windowfeats_subtask_all_y = []

		for p in xrange(0, conf.n_participants):
			featfilename = conf.get_window_features_file(p, window_size)
			timesfilename = conf.get_window_times_file(p, window_size)
			if os.path.exists(featfilename) and os.path.exists(timesfilename):
				data = np.load(featfilename).tolist()
				windowfeats_subtask_all.extend(data)
				windowfeats_subtask_all_y.extend([binned_personality[p, 1:]] * len(data))

				times = np.load(timesfilename)[:, 2:]
				ann = annotation[p,1:]

				ids_annotation = np.zeros((len(data), 3), dtype=int) # person, way/shop, half
				ids_annotation[:,0] = p
				ids_annotation[(times[:,1] < ann[0]),1] = conf.time_window_annotation_wayI
				ids_annotation[(times[:,0] > ann[0]) & (times[:,1] < ann[1]),1] = conf.time_window_annotation_shop
				ids_annotation[(times[:,0] > ann[1]),1] = conf.time_window_annotation_wayII
				ids_annotation[:(len(data)/2), 2] = conf.time_window_annotation_halfI
				ids_annotation[(len(data)/2):, 2] = conf.time_window_annotation_halfII

				windowfeats_subtask_ids.extend(ids_annotation.tolist())
			else:
				print ('did not find ', featfilename)
				sys.exit(1)

		ids = np.array(windowfeats_subtask_ids)
		x = np.array(windowfeats_subtask_all, dtype=float)
		y = np.array(windowfeats_subtask_all_y)
		f1, f2, f3 = conf.get_merged_feature_files(window_size)

		np.savetxt(f1, x, delimiter=',', header=','.join(gs.full_long_label_list), comments='')
		np.savetxt(f2, y, delimiter=',', header=','.join(trait_labels), comments='')
		np.savetxt(f3, ids, delimiter=',', header='Participant ID', comments='')
