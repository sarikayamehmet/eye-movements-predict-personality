import numpy as np
import matplotlib.pyplot as plt
from config import conf
import os, sys
import pandas as pns
from config import names as gs
import getopt
import matplotlib.gridspec as gridspec
from sklearn.metrics import f1_score, accuracy_score

import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
sns.set_context('poster')

dark_color = sns.xkcd_rgb['charcoal grey']
light_color = sns.xkcd_rgb['cloudy blue']

max_n_feat = conf.max_n_feat
m_iter = conf.max_n_iter

featurelabels = gs.full_long_label_list
participant_ids = np.arange(0, conf.n_participants)


def plot_overview():
	all_baselines.groupby(by=['trait', 'clf_name'])['F1'].mean().to_csv(conf.figure_folder +
							'/figure1.csv')
	print 'Figure1.csv written'

	sns.set(font_scale=2.1)
	plt.figure(figsize=(20, 10))
	ax = plt.subplot(1,1,1)
	sns.barplot(x='trait', y='F1', hue='clf_name', data=all_baselines, capsize=.05, errwidth=3,
	            linewidth=3, estimator=np.mean, edgecolor=dark_color,
	            palette={'our classifier': sns.xkcd_rgb['windows blue'],
	                     'most frequent class': sns.xkcd_rgb['faded green'],
	                     'random guess':sns.xkcd_rgb['greyish brown'],
						 'label permutation':sns.xkcd_rgb['dusky pink']
                        }
	            )
	plt.plot([-0.5,6.5], [0.33, 0.33], c=dark_color, linestyle='--', linewidth=3, label='theoretical chance level')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend([handles[1], handles[2], handles[3], handles[4], handles[0]], [labels[1], labels[2], labels[3], labels[4], labels[0]], fontsize=20)
	plt.xlabel('')
	plt.ylabel('F1 score', fontsize=20)
	plt.ylim((0, 0.55))
	filename = conf.figure_folder + '/figure1.pdf'
	plt.savefig(filename, bbox_inches='tight')
	plt.close()
	print 'wrote', filename.split('/')[-1]


if __name__ == "__main__":
	# collect F1 scores for classifiers on all data from a file that was written by evaluation_single_context.py
	datapath = conf.get_result_folder(conf.annotation_all) + '/f1s.csv'
	if not os.path.exists(datapath):
		print 'could not find', datapath
		print 'consider (re-)running evaluation_single_context.py'
		sys.exit(1)
	our_classifier = pns.read_csv(datapath)
	our_classifier['clf_name'] = 'our classifier'

	# baseline 1: guess the most frequent class from each training set that was written by train_baseline.py
	datapath = conf.result_folder + '/most_frequ_class_baseline.csv'
	if not os.path.exists(datapath):
		print 'could not find', datapath
		print 'consider (re-)running train_baseline.py'
		sys.exit(1)
	most_frequent_class_df = pns.read_csv(datapath)
	most_frequent_class_df['clf_name'] = 'most frequent class'

	# compute all other baselines ad hoc
	collection = []
	for trait in xrange(0, conf.n_traits):
		# baseline 2: random guess
		truth = np.genfromtxt(conf.binned_personality_file, skip_header=1, usecols=(trait+1,), delimiter=',')
		for i in xrange(0, 100):
			rand_guess = np.random.randint(1, 4, conf.n_participants)
			f1 = f1_score(truth, rand_guess, average='macro')
			collection.append([f1, conf.medium_traitlabels[trait], i, 'random guess'])

		# baseline 3: label permutation test
		#             was computed using label_permutation_test.sh and written into results. ie. is just loaded here
		for si in xrange(0, m_iter):
			filename_rand = conf.get_result_filename(conf.annotation_all, trait, True, si, add_suffix=True)
			if os.path.exists(filename_rand):
				data = np.load(filename_rand)
				pr = data['predictions']
				dt = truth[pr > 0]
				pr = pr[pr > 0]
				f1 = f1_score(dt, pr, average='macro')
				collection.append([f1, conf.medium_traitlabels[trait], si, 'label permutation'])
			else:
				print 'did not find', filename_rand
				print 'consider (re-)running label_permutation_test.sh'
				sys.exit(1)

	collectiondf = pns.DataFrame(data=collection,columns=['F1','trait','iteration','clf_name'])
	all_baselines = pns.concat([our_classifier, most_frequent_class_df, collectiondf])

	plot_overview()  # Figure 1
