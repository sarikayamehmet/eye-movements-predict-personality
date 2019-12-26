import numpy as np

# global parameters
n_participants = 42
n_traits = 7
max_n_feat = 207
max_n_iter = 100
all_window_sizes = [5, 15, 30, 45, 60, 75, 90, 105, 120, 135]
all_shop_window_sizes = [5, 15]  # at least 3/4 of the people have a time window in these times

# cross validation paramters
n_inner_folds = 3
n_outer_folds = 5

# Random Forest Parameters
tree_max_features = 15
tree_max_depth = 5
n_estimators = 100
max_n_jobs = 5

# given a window size, determine step size correctly for even and odd numbers
def get_step_size(window_size):
	step_size = window_size / 2.0
	if step_size * 10 % 2 == 0:
		step_size = int(step_size)
	return step_size

# relative paths
data_folder = 'data'
info_folder = 'info'
feature_folder = 'features'
result_folder = 'results'
figure_folder = 'figures'
annotation_path = info_folder + '/annotation.csv'
binned_personality_file = info_folder + '/binned_personality.csv'
personality_sex_age_file = info_folder + '/personality_sex_age.csv'

# load the personality trait names from file and map them to abbreviations
traitlabels = np.loadtxt(binned_personality_file, delimiter=',', dtype=str)[0, 1:]
def get_abbr(s):
	return ''.join(item[0] for item in s.split() if item[0].isupper())
medium_traitlabels = [get_abbr(s) if (" " in s) else s for s in traitlabels]
short_traitlabels = [''.join(item[0] for item in tl.split() if item[0].isupper()) for tl in traitlabels]


# dynamically create relative paths for result files to create
def get_result_folder(annotation_val):
	return result_folder + '/A' + str(annotation_val)

def get_result_filename(annotation_val, trait, shuffle_labels, i, add_suffix=False):
	filename = get_result_folder(annotation_val) + '/' + short_traitlabels[trait]
	if shuffle_labels:
		filename += '_rnd'
	filename += '_' + str(i).zfill(3)
	if add_suffix:
		filename += '.npz'
	return filename

def get_feature_folder(participant):
	return feature_folder + '/Participant' + str(participant).zfill(2)

def get_merged_feature_files(window_size):
	return feature_folder + '/merged_features_' + str(window_size) + '.csv', feature_folder + '/merged_traits_' + str(window_size) + '.csv', feature_folder + '/merged_ids_' + str(window_size) + '.csv'

def get_data_folder(participant):
	return data_folder + '/Participant' + str(participant).zfill(2)

def get_window_times_file(participant, window_size):
	return get_feature_folder(participant) + "/window_times_" + str(window_size) + '.npy'

def get_window_features_file(participant, window_size):
	return get_feature_folder(participant) + "/window_features_" + str(window_size) + '.npy'

def get_overall_features_file(participant):
	return get_feature_folder(participant) + "/overall_features.npy"


# parameters for fixation/saccade detection
fixation_radius_threshold = 0.025
fixation_duration_threshold = 0.1
saccade_min_velocity = 2
max_saccade_duration = 0.5

# annotation constants (as given as arguments to train_classifier, and as used for file names in result_folder)
annotation_all = 0
annotation_ways = 1
annotation_shop = 2
annotation_values = [annotation_all, annotation_ways, annotation_shop]

# annotations used in merged_ids_* files in the feature_folder
# column 1
time_window_annotation_wayI = 1
time_window_annotation_shop = 2
time_window_annotation_wayII = 3
# column 2
time_window_annotation_halfI = 1
time_window_annotation_halfII = 2
