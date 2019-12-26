import numpy as np
import matplotlib.pyplot as plt
from config import names as gs
from config import conf
import sys
import math
import os


def get_stats():
	annotation_times = np.genfromtxt(conf.annotation_path, delimiter=',', skip_header=1)[:, 1:]
	shop_duration = annotation_times[:, 1] - annotation_times[:, 0]
	print
	print 'Time spent in the shop:'
	print 'MEAN', np.mean(shop_duration/60.), 'min'
	print 'STD', np.std(shop_duration/60.), 'min'


def get_feature_correlations():
	# find the window size that was most frequently chosen
	hist_sum = np.zeros((len(conf.all_window_sizes)), dtype=int)
	for trait in xrange(0, conf.n_traits):
		for si in xrange(0, 100):
			filename = conf.get_result_filename(conf.annotation_all, trait, False, si, add_suffix=True)
			if os.path.exists(filename):
				data = np.load(filename)
				chosen_window_indices = data['chosen_window_indices']
				hist, _ = np.histogram(chosen_window_indices, bins=np.arange(-0.5, len(conf.all_window_sizes), 1))
				hist_sum += hist
			else:
				print 'did not find', filename

	ws = conf.all_window_sizes[np.argmax(hist_sum)]

	# load features for the most frequently chosen time window
	x_file, y_file, id_file = conf.get_merged_feature_files(ws)
	x_ws = np.genfromtxt(x_file, delimiter=',', skip_header=1)
	ids_ws = np.genfromtxt(id_file, delimiter=',', skip_header=1).astype(int)[:,0]
	y = np.genfromtxt(conf.binned_personality_file, skip_header=1, usecols=xrange(1, conf.n_traits+1), delimiter=',')
	y_ws = np.genfromtxt(y_file, delimiter=',', skip_header=1).astype(int)

	# compute average feature per person
	avg_x_ws = np.zeros((conf.n_participants, conf.max_n_feat))
	for p in xrange(0,conf.n_participants):
		avg_x_ws[p,:] = np.mean(x_ws[ids_ws == p, :], axis=0)

	feature_correlations_avg = []
	for fi in xrange(0, conf.max_n_feat):
		C_avg = np.corrcoef(y.transpose(), avg_x_ws[:, fi])[-1][:-1]
		feature_correlations_avg.append(C_avg)

	feature_correlations_avg = np.array(feature_correlations_avg)

	# find the 5th to highest correlation for each trait and write them into a .tex table - see Table 4 in SI
	n = 15
	highest_correlated_features = []
	highest_correlated_features_lists = []
	highest_correlated_features_names = []
	for t in xrange(0, conf.n_traits):
		hcf = feature_correlations_avg[:,t].argsort()[-n:]
		locallist = []
		for f in hcf:
			if f not in highest_correlated_features:
				highest_correlated_features.append(f)
				highest_correlated_features_names.append(gs.full_long_label_list[f].lower())
			locallist.append(f)

		highest_correlated_features_lists.append(locallist)

	features = zip(highest_correlated_features_names, highest_correlated_features)
	highest_correlated_features = [y for (x,y) in sorted(features)]
	#highest_correlated_features.sort()

	filename = conf.figure_folder + '/table4.tex'
	print len(highest_correlated_features)
	with open(filename, 'w') as f:
		f.write('feature&Neur.&Extr.&Open.&Agree.&Consc.&PCS&CEI')
		f.write('\\\\\n\hline\n')
		for fi in highest_correlated_features:
			f.write(gs.full_long_label_list[fi])
			for t in xrange(0, conf.n_traits):
				fc = feature_correlations_avg[fi,t]
				if math.isnan(fc):
					f.write('&-')
				elif fi in highest_correlated_features_lists[t]:
					f.write('&\\textbf{'+'%.2f}'%fc)
				else:
					f.write('&'+'%.2f'%fc)
			f.write('\\\\\n')
	print
	print filename, 'written'


if __name__ == "__main__":
	import os
	if not os.path.exists(conf.figure_folder):
		os.makedirs(conf.figure_folder)
	get_stats()  # prints statistics on the time participants spent inside the shop
	get_feature_correlations()  # Table 4
