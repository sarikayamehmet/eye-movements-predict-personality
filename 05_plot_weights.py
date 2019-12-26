import numpy as np
import matplotlib.pyplot as plt
from config import conf
import os, sys
import pandas as pns
from config import names as gs
import getopt
import matplotlib.gridspec as gridspec
from sklearn.metrics import f1_score

import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
sns.set_context('poster')

dark_color = sns.xkcd_rgb['charcoal grey']
light_color = sns.xkcd_rgb['cloudy blue']

def plot_weights():
	# for each personality trait, compute the list of median feature importances across all cross validation folds and iterations
	medianlist = []
	for t in xrange(0, conf.n_traits):
		medianlist.append(
			list(imp_df.loc[imp_df['T'] == t].groupby(by='feat_num')['feature importance'].median()))

	# find the 5th to highest feature importance for each trait and write their importances into a .tex table - see Table 2, SI
	n = 15
	most_important_features = []
	most_important_features_lists = []
	for ml in medianlist:
		locallist = []
		for i in xrange(1,(n+1)):
			fn = gs.full_long_label_list[int(np.argsort(np.array(ml))[-i])]
			locallist.append(fn)
			if fn not in most_important_features:
				most_important_features.append(fn)
		most_important_features_lists.append(locallist)
	most_important_features.sort()

	# write the full list of feature importances into a .tex table - shown in Table 2, SI
	filename = conf.figure_folder + '/table2.tex'
	with open(filename, 'w') as f:
		f.write('feature&Neur.&Extr.&Open.&Agree.&Consc.&PCS&CEI')
		f.write('\\\\\n\hline\n')
		for fi in xrange(0, len(most_important_features)):
			f.write(most_important_features[fi])
			for t in xrange(0, conf.n_traits):
				m = imp_df[(imp_df['T'] == t)&(imp_df.feature == most_important_features[fi])]['feature importance'].median()
				if most_important_features[fi] in most_important_features_lists[t]:
					f.write('& \\textbf{' + '%.3f}' % m)
				else:
					f.write('&' + '%.3f' % m)
			f.write('\\\\\n')
	print filename, 'written.'

	# create Figure 2
	# first collect the set of individual top TOP_N features per trait:
	TOP_N = 10
	featlabels = []
	for trait in xrange(0, conf.n_traits):
		basedata = imp_df.loc[imp_df['T'] == trait]
		gp = basedata.groupby(by='feature')['feature importance'].median()
		order = gp.sort_values(ascending=False)
		featlabels.extend(order[:TOP_N].keys())
	super_feats = np.unique(np.array(featlabels))

	# collect the sum of feature importances for these labels, to sort the features by their median
	super_feats_importance_sum = np.zeros((len(super_feats)))
	for i in xrange(0, len(super_feats)):
		super_feats_importance_sum[i] = imp_df[imp_df.feature==super_feats[i]].groupby(by=['T'])['feature importance'].median().sum()
	super_feats_sort_indices = np.argsort(super_feats_importance_sum)[::-1]

	# add some interesting features from related work to the list of features whose importance will be shown
	must_have_feats = [
		'inter quartile range x', 'range x', 'maximum x', 'std x', '1st quartile x', 'range pupil diameter', 'median y',
		'mean difference of subsequent x', 'mean fixation duration', '3rd quartile y',
		'fixation rate', 'mean saccade amplitude', 'dwelling time'
	]
	# but only add them if they are not in the list yet
	additional_feats = np.array([a for a in must_have_feats if a not in super_feats], dtype=object)

	# collect the sum of feature importances for these labels as well, so they can be sorted by their median importance in the plot
	additional_feats_importance_sum = np.zeros((len(additional_feats)))
	for trait in xrange(0, conf.n_traits):
		basedata = imp_df.loc[imp_df['T'] == trait]
		for i in xrange(0, len(additional_feats)):
			logi = basedata.feature == additional_feats[i]
			additional_feats_importance_sum[i] += float(basedata[logi]['feature importance'].median())
	additional_feats_sort_indices = np.argsort(additional_feats_importance_sum)[::-1]

	# create the figure
	plt.figure(figsize=(20, 12))
	grs = gridspec.GridSpec(len(super_feats) + len(additional_feats) + 1, conf.n_traits)

	for trait in xrange(0, conf.n_traits):
		# upper part of the figure, i.e. important features
		ax = plt.subplot(grs[:len(super_feats),trait])
		basedata = imp_df.loc[imp_df['T'] == trait]
		feat_importances = []
		for i in xrange(0, len(super_feats)):
			logi = basedata.feature == super_feats[super_feats_sort_indices][i]
			feat_importances.append(list(basedata[logi]['feature importance']))
		bp = plt.boxplot(x=feat_importances, #notch=True, labels=super_feats[super_feats_sort_indices],
		                 patch_artist=True, sym='', vert=False, whis='range', positions=np.arange(0,len(feat_importances)))

		# asthetics
		for i in xrange(0, len(super_feats)):
			bp['boxes'][i].set(color=dark_color)
			bp['boxes'][i].set(facecolor=light_color)
			bp['whiskers'][2 * i].set(color=dark_color, linestyle='-')
			bp['whiskers'][2 * i + 1].set(color=dark_color, linestyle='-')
			bp['caps'][2 * i].set(color=dark_color)
			bp['caps'][2 * i + 1].set(color=dark_color)
			bp['medians'][i].set(color=dark_color)

		if not trait == 0:
			plt.ylabel('')
			plt.setp(ax.get_yticklabels(), visible=False)
		else:
			ax.set_yticklabels(super_feats[super_feats_sort_indices])

		xlimmax = 0.47
		xticks = [0.15, 0.35]
		plt.xlim((0, xlimmax))
		plt.xticks(xticks)
		plt.setp(ax.get_xticklabels(), visible=False)

		# lower part of the figure, i.e. features from related work
		ax = plt.subplot(grs[(-len(additional_feats)):, trait])
		basedata = imp_df.loc[imp_df['T'] == trait]
		feat_importances = []
		for i in xrange(0, len(additional_feats)):
			logi = basedata.feature == additional_feats[additional_feats_sort_indices][i]
			feat_importances.append(basedata[logi]['feature importance'])
		bp = plt.boxplot(x=feat_importances, patch_artist=True, sym='', vert=False, whis='range',
		                 positions=np.arange(0,len(feat_importances)))

		# asthetics
		for i in xrange(0, len(additional_feats)):
			bp['boxes'][i].set(color=dark_color)
			bp['boxes'][i].set(facecolor=light_color) #, alpha=0.5)
			bp['whiskers'][2 * i].set(color=dark_color, linestyle='-')
			bp['whiskers'][2 * i + 1].set(color=dark_color, linestyle='-')
			bp['caps'][2 * i].set(color=dark_color)
			bp['caps'][2 * i + 1].set(color=dark_color)
			bp['medians'][i].set(color=dark_color) #, linewidth=.1)

		if not trait == 0:
			plt.ylabel('')
			plt.setp(ax.get_yticklabels(), visible=False)
		else:
			ax.set_yticklabels(additional_feats[additional_feats_sort_indices])
		plt.xlim((0, xlimmax))
		plt.xticks(xticks)
		if trait == 3:
			plt.xlabel(conf.medium_traitlabels[trait] + '\n\nFeature Importance')
		else:
			plt.xlabel(conf.medium_traitlabels[trait])

	filename = conf.figure_folder + '/figure2.pdf'
	plt.savefig(filename, bbox_inches='tight')
	print filename.split('/')[-1], 'written.'
	plt.close()


if __name__ == "__main__":
	# target file names - save table of F1 scores, feature importances and majority predictions there
	datapathI = conf.get_result_folder(conf.annotation_all) + '/f1s.csv'  # F1 scores from each iteration
	datapathII = conf.get_result_folder(conf.annotation_all) + '/feature_importance.csv'  # Feature importance from each iteration
	datapathIII = conf.get_result_folder(conf.annotation_all) + '/majority_predictions.csv'  # Majority voting result for each participant over all iterations

	if not os.path.exists(conf.figure_folder):
		os.mkdir(conf.figure_folder)

	# if target files do not exist yet, create them
	if (not os.path.exists(datapathI)) or (not os.path.exists(datapathII)) or (not os.path.exists(datapathIII)):
		f1s = []
		feature_importances = []
		majority_predictions = []
		for trait in xrange(0, conf.n_traits):
			predictions = np.zeros((conf.n_participants, conf.max_n_iter),dtype=int)-1
			ground_truth = np.loadtxt(conf.binned_personality_file, delimiter=',', skiprows=1, usecols=(trait+1,))
			for si in xrange(0, conf.max_n_iter):
				filename = conf.get_result_filename(conf.annotation_all, trait, False, si, add_suffix=True)
				if os.path.exists(filename):
					data = np.load(filename)
					if (data['predictions'] > 0).all():
						assert data['f1'] == f1_score(ground_truth, data['predictions'], average='macro')
						f1s.append([data['f1'], conf.medium_traitlabels[trait]])
					else:
						#   if there was no time window for a condition, like if shopping data only is evaluated,
						#   the F1 score for each person without a single time window will be set to -1
						#   but should not be used as such to compute the mean F1 score.
						#   Thus, here the F1 score is re-computed on the relevant participants only.
						pr = data['predictions']
						pr = pr[pr > 0]

						dt = ground_truth[pr > 0]

						f1s.append([f1_score(dt, pr, average='macro'), conf.medium_traitlabels[trait]])

					for outer_cv_i in xrange(0, 5):  # number outer CV, not person anymore
						for fi in xrange(0, conf.max_n_feat):
							feature_importances.append([data['feature_importances'][outer_cv_i, fi], trait, gs.full_long_label_list[fi], fi])

					predictions[:,si] = data['predictions']
				else:
					print 'did not find', filename

			# compute majority voting for each participant over all iterations
			for p in xrange(0, conf.n_participants):
				(values, counts) = np.unique(predictions[p, predictions[p,:]>0], return_counts=True)
				ind = np.argmax(counts)
				majority_predictions.append([values[ind], p, conf.medium_traitlabels[trait]])

		f1s_df = pns.DataFrame(data=f1s, columns=['F1', 'trait'])
		f1s_df.to_csv(datapathI)

		imp_df = pns.DataFrame(data=feature_importances, columns=['feature importance', 'T', 'feature', 'feat_num'])
		imp_df.to_csv(datapathII)

		majority_predictions_df = pns.DataFrame(data=majority_predictions, columns=['prediction','participant','trait'])
		majority_predictions_df.to_csv(datapathIII)

	else:
		print 'No new results are collected as previous results were available. If you want to overwrite them, please delete the following files:'
		print datapathI
		print datapathII
		print datapathIII

	f1s_df = pns.read_csv(datapathI)
	imp_df = pns.read_csv(datapathII)
	majority_predictions_df = pns.read_csv(datapathIII)

	plot_weights()  # Figure 2
