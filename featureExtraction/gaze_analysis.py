#!/usr/bin/python
import numpy as np
import sys, os
from featureExtraction import event_detection as ed
import operator
from config import names as gs

class gazeAnalysis (object):
	# dictionary for saccade-based n-grams:
	# each character encodes one direction, capital characters stand for long saccades, the others for short ones
	# short means the saccade amplitude is less than 2 fixation_radius_thresholds
	#						U
 	#					O		A
 	#				N		u		B
	#			M		n		b		C
	#		L		l		.		r		R
	#			K		j		f		E
	#				J		d		F
	#					H		G
	#						D
	sacc_dictionary = ['A', 'B', 'C', 'R', 'E', 'F', 'G', 'D', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'U', 'u', 'b', 'r', 'f',
						'd', 'j', 'l', 'n']
	sacc_bins_two = [a+b for a in sacc_dictionary for b in sacc_dictionary]
	sacc_bins_three = [a+b+c for a in sacc_dictionary for b in sacc_dictionary for c in sacc_dictionary]
	sacc_bins_four = [a+b+c+d for a in sacc_dictionary for b in sacc_dictionary for c in sacc_dictionary for d in sacc_dictionary]
	sacc_bins = [sacc_dictionary, sacc_bins_two, sacc_bins_three, sacc_bins_four]

	# dictionary for saccade and fixation-based n-grams:
	# S are saccades, long or short (i.e. longer or shorter than the fixation radius), and up/down/right/left
	# e.g. S_lu is a long saccade up
	# F are fixations, either long or short (i.e. longer or shorter than twice the minimum fixation duration)
	# saccFix_dictionary = ['S_lu', 'S_ld', 'S_lr', 'S_ll', 'S_su', 'S_sd', 'S_sr', 'S_sl', 'F_l', 'F_s']
	saccFix_dictionary = ['U', 'D', 'R', 'L', 'u', 'd', 'r', 'l', 'F', 'f']
	saccFix_bins_two = [a+b for a in saccFix_dictionary for b in saccFix_dictionary]
	saccFix_bins_three = [a+b+c for a in saccFix_dictionary for b in saccFix_dictionary for c in saccFix_dictionary]
	saccFix_bins_four = [a+b+c+d for a in saccFix_dictionary for b in saccFix_dictionary for c in saccFix_dictionary for d in saccFix_dictionary]
	saccFix_bins = [saccFix_dictionary, saccFix_bins_two, saccFix_bins_three, saccFix_bins_four]

	def __init__(self, gaze, fixation_radius_threshold, fixation_duration_threshold, saccade_min_velocity,max_saccade_duration,
					pupil_diameter=None, event_strings=None, ti=0, xi=1, yi=2):
		assert gaze.size > 0

		# save data in instance
		self.gaze = gaze
		self.diams = pupil_diameter
		self.event_strings = event_strings

		# save constants, indices and thresholds that will be used muttiple times
		self.fixation_radius_threshold = fixation_radius_threshold
		self.fixation_duration_threshold = fixation_duration_threshold
		self.xi = xi
		self.yi = yi
		self.ti = ti

		# detect errors, fixations, saccades and blinks
		self.errors = self.detect_errors()
		self.fixations, self.saccades, self.blinks, self.wordbook_string = \
			ed.detect_all(self.gaze, self.errors, self.ti, self.xi, self.yi, pupil_diameter=pupil_diameter,
							event_strings=event_strings, fixation_duration_threshold=fixation_duration_threshold,
							fixation_radius_threshold=fixation_radius_threshold, saccade_min_velocity=saccade_min_velocity,
							max_saccade_duration=max_saccade_duration)

	def detect_errors(self, confidence_threshold=0.8, outlier_threshold=0.5):
		"""
		:param confidence_threshold: threshold below which all gaze data is deleted
		:param outlier_threshold: threshold beyond which gaze must not be outside the calibration area (i.e. [0,1])
		"""
		errors = np.full((len(self.gaze)), False, dtype=bool)

		# gaze is nan
		errors[np.isnan(self.gaze[:, self.xi])] = True
		errors[np.isnan(self.gaze[:, self.yi])] = True

		# gaze outside a certain range
		errors[self.gaze[:, self.xi] < -outlier_threshold] = True
		errors[self.gaze[:, self.xi] > outlier_threshold + 1] = True

		errors[self.gaze[:, self.yi] < -outlier_threshold] = True
		errors[self.gaze[:, self.yi] > outlier_threshold + 1] = True

		return errors

	def get_window_features(self, sliding_window_size, sliding_window_step_size,  start_index=-1, end_index=-1):
		"""
		computes features using a sliding window approach with the given sliding_window_size and sliding_window_step_size
		"""
		# if no start and end index are given, use all data
		if start_index == -1 and end_index == -1:
			start_index = 0
			end_index = len(self.gaze[:, 0]) - 1

		# compute start and end times of each resulting sliding window
		window_times = self.get_sliding_windows(start_index, end_index, sliding_window_size, sliding_window_step_size)

		#compute features for each of these windows:
		window_feature_list = []
		for [a,b,at,bt] in window_times:
			overallstats = self.get_full_feature_vector(a, b)
			window_feature_list.append(overallstats)
		assert len(gs.full_label_list) == len(window_feature_list[0])

		window_feature_list = np.array(window_feature_list)
		window_times = np.array(window_times)
		return window_feature_list, window_times

	def get_full_feature_vector(self, start_index, end_index):
		"""
		assembles the full feature vector of its part:
		features based on fixations/saccades/blinks, raw data, heatmaps, n-grams based on saccades and n-grams based on saccades and fixations
		"""
		# features based on events, i.e. fixations/saccades/blinks
		features = self.get_event_features(start_index, end_index)
		assert len(gs.event_feature_labels) == len(features)

		# features based on raw data, like quartiles of gaze posiitons
		raw_features = self.get_raw_features(start_index, end_index)
		features.extend(raw_features)
		assert len(gs.position_feature_labels) == len(raw_features)

		# heatmap features
		heatmap_features = self.get_heatmap_features(start_index, end_index)
		features.extend(heatmap_features)
		assert len(gs.heatmap_feature_labels) == len(heatmap_features)

		# n-gram features based on saccades
		sacc_wordbook_features = self.get_sacc_ngram_features(start_index, end_index)
		features.extend(sacc_wordbook_features)
		assert len(gs.get_wordbook_feature_labels('')) == len(sacc_wordbook_features)

		# n-gram features based on saccades and fixations
		saccFix_wordbook_features = self.get_saccFix_ngram_features(start_index, end_index)
		features.extend(saccFix_wordbook_features)
		assert len(gs.get_wordbook_feature_labels('')) == len(saccFix_wordbook_features)

		return features

	def get_event_features(self, start_index, end_index):
		"""
		computes features based on fixations, saccades and blinks within the selected gaze window
		"""
		event_features = []

		# get non-errouneous samples between start_index and end_index:
		x,y,d = self.get_x_y_d_part(start_index, end_index)

		data = self.gaze[start_index:(end_index+1), :]
		fixations, saccades, wordbook_string, blinks = self.get_FixSacWb_timed(start_index, end_index)

		n, m = data.shape
		timespan = (data[-1,self.ti] - data[0,self.ti])

		num_fix =len(fixations)
		num_sacc =len(saccades)

		if timespan == 0:
			fix_rate = 0
			sacc_rate = 0
		else:
			fix_rate = num_fix / float(timespan)
			sacc_rate = num_sacc / float(timespan)

		event_features.append(fix_rate) # 0. rate of fixations
		event_features.append(sacc_rate) # 1. rate of saccades


		sacc_UR_L = wordbook_string.count('U') + wordbook_string.count('A')  + wordbook_string.count('B') + wordbook_string.count('C')
		sacc_BR_L = wordbook_string.count('R') + \
							wordbook_string.count('E')  + \
							wordbook_string.count('F')  + \
							wordbook_string.count('G')
		sacc_UL_L = wordbook_string.count('O')  \
							+ wordbook_string.count('N')  \
							+ wordbook_string.count('M')  \
							+ wordbook_string.count('L')
		sacc_BL_L = wordbook_string.count('K') \
							+ wordbook_string.count('J') \
							+ wordbook_string.count('H') \
							+ wordbook_string.count('D')

		sacc_UR_S = wordbook_string.count('u') \
							+ wordbook_string.count('b')
		sacc_BR_S = wordbook_string.count('r') \
							+ wordbook_string.count('f')
		sacc_UL_S = wordbook_string.count('n') \
							+ wordbook_string.count('l')
		sacc_BL_S = wordbook_string.count('j') \
							+ wordbook_string.count('d')

		num_s_sacc = sacc_UR_S + sacc_BR_S + sacc_UL_S + sacc_BL_S
		num_la_sacc = sacc_UR_L + sacc_BR_L + sacc_UL_L + sacc_BL_L
		num_r_sacc = sacc_UR_S + sacc_BR_S + sacc_UR_L + sacc_BR_L
		num_l_sacc = sacc_UL_S + sacc_BL_S + sacc_UL_L + sacc_BL_L

		if timespan > 0:
			event_features.append(num_s_sacc / float(timespan)) #2. rate of small saccades
			event_features.append(num_la_sacc / float(timespan)) #3. rate of large saccades
			event_features.append(num_r_sacc / float(timespan)) #4. rate of pos saccades
			event_features.append(num_l_sacc / float(timespan)) #5. rate of neg saccades
		else:
			event_features.extend([0]*4)

		if num_fix > 0:
			event_features.append(num_sacc / float(num_fix)) #	6. ratio saccades / fixations
		else:
			event_features.append(0)

		if num_sacc > 0:
			event_features.append(num_s_sacc /float(num_sacc)) # 	7. ratio small sacc
			event_features.append(num_la_sacc / float(num_sacc)) #	8. ratio large sacc
			event_features.append(num_r_sacc / float(num_sacc)) #	9. ratio pos sacc
			event_features.append(num_l_sacc / float(num_sacc)) #	10. ratio neg sacc
		else:
			event_features.extend([0]*4)

		sacc_array = np.array(saccades)
		fix_array = np.array(fixations)

		if sacc_array.size > 0:
			# amplitude features
			amplitudes = sacc_array[:, gs.sacc_amplitude_i]
			event_features.append(np.mean(amplitudes))  # 11: mean sacc amplitude
			event_features.append(np.var(amplitudes))  # 12: var  sacc amplitude
			event_features.append(amplitudes.min())    # 13 min sacc amplitude
			event_features.append(amplitudes.max())  # 14: max  sacc amplitude

			# peak velocity features
			velocities = sacc_array[:, gs.sacc_peak_vel_i]
			event_features.append(np.mean(velocities))  # 15: mean peak velocity
			event_features.append(np.var(velocities))  # 16: var peak velocity
			event_features.append(velocities.min())   # 17: min  peak velocity
			event_features.append(velocities.max())  # 18: max  peak velocity

			if sacc_array[0, :].size == 13:
				event_features.append(np.mean(sacc_array[:, gs.sacc_mean_diam_i]))  # 19 mean mean diameter
				event_features.append(np.var(sacc_array[:, gs.sacc_mean_diam_i]))  # 20 var mean diameter
				event_features.append(np.mean(sacc_array[:, gs.sacc_var_diam_i]))  # 21 mean var diameter
				event_features.append(np.var(sacc_array[:, gs.sacc_var_diam_i]))  # 22 var var diameter
			else:
				event_features.extend([0]*4)
		else:
			event_features.extend([0]*12)

		if fix_array.size > 0:
			durations = np.array(fix_array[:, gs.fix_end_t_i]) - np.array(fix_array[:, gs.fix_start_t_i])
			event_features.append(np.mean(durations))  # 23: mean fix duration
			event_features.append(np.var(durations))  # 24: var  fix duration
			event_features.append(durations.min())  # 25: min  fix duration
			event_features.append(durations.max())  # 26: max  fix duration
			event_features.append(durations.sum())  # 27: dwelling time

			event_features.append(np.mean(fix_array[:, gs.fix_mean_succ_angles]))  # 28: mean mean subsequent angle
			event_features.append(np.var(fix_array[:, gs.fix_mean_succ_angles]))  # 28: var mean subsequent angle
			event_features.append(np.mean(fix_array[:, gs.fix_var_succ_angles]))  # 28: mean var subsequent angle
			event_features.append(np.var(fix_array[:, gs.fix_var_succ_angles]))  # 28: var var subsequent angle

			lnnII = np.logical_not(np.isnan(fix_array[:, gs.fix_var_x_i]))
			lnnIII = np.logical_not(np.isnan(fix_array[:, gs.fix_var_y_i]))
			event_features.append(np.mean(fix_array[lnnII, gs.fix_var_x_i]))  # mean var x
			event_features.append(np.mean(fix_array[lnnIII, gs.fix_var_y_i]))  # mean var y
			event_features.append(np.var(fix_array[lnnII, gs.fix_var_x_i]))  # 29: var var x
			event_features.append(np.var(fix_array[lnnIII, gs.fix_var_y_i]))  # 30: var var y

			if fix_array[0, :].size == 12:
				event_features.append(np.mean(fix_array[:, gs.fix_mean_diam_i]))  # 31 mean mean diameter
				event_features.append(np.var(fix_array[:, gs.fix_mean_diam_i]))  # 32 var mean diameter
				event_features.append(np.mean(fix_array[:, gs.fix_var_diam_i]))  # 33 mean var diameter
				event_features.append(np.var(fix_array[:, gs.fix_var_diam_i]))  # 34 var var diameter
			else:
				event_features.extend([0]*4)
		else:
			event_features.extend([0]*17)

		blink_array = np.array(blinks)
		if blink_array.size > 0:
			durations = np.array(blink_array[:, 1]) - np.array(blink_array[:, 0])
			event_features.append(np.mean(durations)) #35
			event_features.append(np.var(durations)) #36
			event_features.append(durations.min()) #37
			event_features.append(durations.max()) #38
			event_features.append(np.true_divide(len(blink_array), timespan)) #39
		else:
			event_features.extend([0]*5)
		return event_features

	def get_x_y_d_part(self, start_index, end_index):
		x = self.gaze[start_index:(end_index + 1), self.xi]
		y = self.gaze[start_index:(end_index + 1), self.yi]
		d = (self.diams[start_index:(end_index + 1), 1] + self.diams[start_index:(end_index + 1), 2]) / 2.0

		err = self.errors[start_index:(end_index + 1)]

		x = x[np.logical_not(err)]
		y = y[np.logical_not(err)]
		d = d[np.logical_not(err)]
		return x,y,d

	def get_raw_features(self, start_index, end_index):
		"""
		computes features based on raw gaze data, like percentiles of x coordinates
		"""
		raw_features = []

		x,y,d = self.get_x_y_d_part(start_index, end_index)

		for a in [x, y, d]:
			raw_features.append(np.mean(a))
		for a in [x, y, d]:
			raw_features.append(np.amin(a))
		for a in [x, y, d]:
			raw_features.append(np.amax(a))
		for a in [x, y, d]:
			raw_features.append(np.amax(a) - np.amin(a))
		for a in [x, y, d]:
			raw_features.append(np.std(a))
		for a in [x, y, d]:
			raw_features.append(np.median(a))
		for a in [x, y, d]:
			raw_features.append(np.percentile(a, 25))
		for a in [x, y, d]:
			raw_features.append(np.percentile(a, 75))
		for a in [x, y, d]:
			raw_features.append(np.percentile(a, 75) - np.percentile(a, 25))
		for a in [x, y, d]:
			raw_features.append(np.mean(np.abs(a[1:] - a[:-1])))
		for a in [x, y, d]:
			raw_features.append(np.mean(a[1:] - a[:-1]))

		dx = x[:-1] - x[1:]
		dy = y[:-1] - y[1:]
		succ_angles = np.arctan2(dy, dx)
		raw_features.append(np.mean(succ_angles))#  28: mean subsequent angle
		return raw_features

	def get_heatmap_features(self, start_index, end_index):
		"""
		computes a heatmap over the raw gaze positions, in a 8x8 grid
		"""
		x,y,d = self.get_x_y_d_part(start_index, end_index)

		xmin = np.percentile(x, 2.5)
		xmax = np.percentile(x, 97.5)
		ymin = np.percentile(y, 2.5)
		ymax = np.percentile(y, 97.5)

		heatmap, xedges, yedges = np.histogram2d(x, y, bins=(8, 8), range=[[xmin, xmax], [ymin, ymax]])
		normalised_flat_heatmap = heatmap.flatten() / np.sum(heatmap)
		return normalised_flat_heatmap

	def get_sacc_ngram_features(self, start_index, end_index):
		# find those saccades that either start or end within start_index and end_index, or start before start_index and end after end_index
		mysacc = [sacc for sacc in self.saccades if (sacc[gs.sacc_start_index_i] > start_index and sacc[gs.sacc_start_index_i] < end_index)
													or (sacc[gs.sacc_end_index_i] > start_index and sacc[gs.sacc_end_index_i] < end_index)
													or (sacc[gs.sacc_start_index_i] < start_index and sacc[gs.sacc_end_index_i] > end_index)]
		# create string representing all saccades in mysacc
		mywbs = [self.wordbook_string[mysacc.index(sacc)] for sacc in mysacc]

		# create all possible n-grams of a certain length
		ngrams = []
		ngrams.append(mywbs)
		for n in xrange(2,5):
			ngrams.append([reduce(operator.add, mywbs[i:i+n]) for i in range(len(mywbs) - n)])

		# compute histograms of the actual n-grams occuring in the data
		histograms=[]
		for i in xrange(0,4):
			histograms.append(dict((x, ngrams[i].count(x)) for x in self.sacc_bins[i]))

		# compute features from each histogram and append them to one list
		sacc_ngram_features = []
		for h in histograms:
			wb_feat = self.get_ngram_features(h)
			sacc_ngram_features.extend(wb_feat)

		return sacc_ngram_features

	def get_saccFix_wb_string_saccades(self, sacc):
		"""
		returns a string for a single saccade that will be used for n-gram features based on saccades and fixations
		"""
		amplitude = sacc[gs.sacc_amplitude_i]
		angle_rad = sacc[gs.sacc_angle_i]
		angle_deg = np.true_divide(angle_rad * 180.0, np.pi)

		# 0 degrees is pointing to the right
		if angle_deg < 45:
			wb_str = 'r'
		elif angle_deg < 135:
			wb_str = 'u'
		elif angle_deg < 225:
			wb_str = 'l'
		elif angle_deg < 315:
			wb_str = 'd'
		else:
			wb_str = 'r'

		if amplitude >= 2 * self.fixation_radius_threshold:  # less than 2 fixation_radius_thresholds
			wb_str = wb_str.upper()

		return wb_str

	def get_saccFix_wb_string_fixations(self, fix):
		"""
		returns a string for a single fixation that will be used for n-gram features based on saccades and fixations
		"""
		if fix[gs.fix_end_t_i] - fix[gs.fix_start_t_i] < 2 * self.fixation_duration_threshold:
			return 'f'
		else:
			return 'F'

	def get_saccFix_ngram_features(self, start_index, end_index):
		"""
		computes n-gram features based on saccades and fixations
		"""
		# find all saccades and fixations between start_index and end_index,
		# and create a string of their encodings
		sacc_index = 0
		fix_index = 0
		wordbook_string = []

		sacc_start_i = gs.sacc_start_index_i
		sacc_end_i = gs.sacc_end_index_i
		fix_start_i = gs.fix_start_index_i
		fix_end_i = gs.fix_end_index_i

		while (self.saccades[sacc_index][sacc_end_i] < start_index) and (sacc_index < len(self.saccades) - 1):
			sacc_index += 1

		while (self.fixations[fix_index][fix_end_i] < start_index) and (fix_index < len(self.fixations) - 1):
			fix_index += 1

		while sacc_index < len(self.saccades) and fix_index < len(self.fixations):
			if (self.saccades[sacc_index][sacc_start_i] < end_index) and (self.fixations[fix_index][fix_start_i] < end_index):
				if self.saccades[sacc_index][sacc_start_i] < self.fixations[fix_index][fix_start_i]:
					wordbook_string.append(self.get_saccFix_wb_string_saccades(self.saccades[sacc_index]))
					sacc_index += 1
				else:
					wordbook_string.append(self.get_saccFix_wb_string_fixations(self.fixations[fix_index]))
					fix_index += 1
			elif self.saccades[sacc_index][sacc_start_i] < end_index:
				wordbook_string.append(self.get_saccFix_wb_string_saccades(self.saccades[sacc_index]))
				sacc_index += 1
			elif self.fixations[fix_index][fix_start_i] < end_index:
				wordbook_string.append(self.get_saccFix_wb_string_fixations(self.fixations[fix_index]))
				fix_index += 1
			else:
				sacc_index += 1
				fix_index += 1

		# compute all possible n-grams
		ngrams = []
		ngrams.append(wordbook_string)
		for n in xrange(2,5):
			ngrams.append([reduce(operator.add, wordbook_string[i:i+n]) for i in range(len(wordbook_string) - n)])

		# compute histograms for n-grams
		histograms=[]
		for i in xrange(0,4):
			histograms.append(dict((x, ngrams[i].count(x)) for x in self.saccFix_bins[i]))

		# compute features from each histogram and append to one list
		ngram_features = []
		for h in histograms:
			wb_feat = self.get_ngram_features(h)
			ngram_features.extend(wb_feat)

		return ngram_features

	def get_FixSacWb_timed(self, start_index, end_index):
		"""
		returns list of fixations, saccades, blinks and the associated wordbook_string
		for the time between start_index and end_index
		"""
		myfix = [fix for fix in self.fixations if (fix[gs. fix_start_index_i] > start_index
		                                           and fix[gs. fix_start_index_i] < end_index)
		         or (fix[gs.fix_end_index_i]>start_index and fix[gs.fix_end_index_i]<end_index)
		         or (fix[gs.fix_start_index_i]<start_index and fix[gs.fix_end_index_i]>end_index)]

		mysacc = [sacc for sacc in self.saccades if (sacc[gs.sacc_start_index_i]>start_index and sacc[gs.sacc_start_index_i]<end_index)
		          or (sacc[gs.sacc_end_index_i]>start_index and sacc[gs.sacc_end_index_i]<end_index)
		          or (sacc[gs.sacc_start_index_i]<start_index and sacc[gs.sacc_end_index_i]>end_index)]

		mywbs = [self.wordbook_string[self.saccades.index(sacc)] for sacc in mysacc]
		blinks = [b for b in self.blinks if (b[gs.blink_start_index_i]>start_index and b[gs.blink_start_index_i]<end_index)
		          or (b[gs.blink_end_index_i]>start_index and b[gs.blink_end_index_i]<end_index)
		          or (b[gs.blink_start_index_i]<start_index and b[gs.blink_end_index_i]>end_index)]

		return myfix, mysacc, mywbs, blinks

	def get_sliding_windows(self, start_index, end_index, sliding_window_size, sliding_window_step_size):
		"""
		computes a list of time windows resulting from the sliding windows approach with the given sliding_window_size and sliding_window_step_size
		"""
		window_times = []  # consisting of lists [a,b,t_a,t_b] where a is start index, b ist end index

		a = start_index
		b = min(self.getEndWindow(a, sliding_window_size), end_index)
		if self.check_sliding_window_conditions(a, b, end_index, sliding_window_size):
			window_times.append([a, b, self.gaze[a, self.ti], self.gaze[b, self.ti]])

		while b < end_index:
			a = self.getStartWindow(a, sliding_window_step_size)
			b = self.getEndWindow(a, sliding_window_size)
			# check if some conditions are fulfilled and if so, add to the window list
			if self.check_sliding_window_conditions(a, b, end_index, sliding_window_size):
				window_times.append([a, b, self.gaze[a, self.ti], self.gaze[b, self.ti]])
		return window_times

	def check_sliding_window_conditions(self, a, b, end_index, sliding_window_size):
		# discard too short or too long sliding windows
		window_duration = self.gaze[b, self.ti] - self.gaze[a, self.ti]
		min_duration = sliding_window_size - 0.1*sliding_window_size
		max_duration = sliding_window_size + 0.1*sliding_window_size
		if window_duration < min_duration or window_duration>max_duration:
			return False

		if b <= end_index:
			errors_in_window = np.sum(self.errors[a:(b+1)]) / (b-a)

			if errors_in_window < 0.5:
				# discard window if no saccade or fixation was detected
				fixations, saccades, wordbook_string, blinks = self.get_FixSacWb_timed(a,b)
				if len(fixations) == 0 and len(saccades) == 0:
					return False

				xgaze = self.gaze[a:(b+1), self.xi][np.logical_not(self.errors[a:(b+1)])]
				ygaze = self.gaze[a:(b+1), self.yi][np.logical_not(self.errors[a:(b+1)])]

				# discard window if less than 5 samples
				if len(xgaze) < 5:
					return False

				# exclude windows with more than 66% constant gaze
				(xvals, xcounts) = np.unique(xgaze, return_counts=True)
				(yvals, ycounts) = np.unique(ygaze, return_counts=True)
				xcounts = xcounts / float(np.sum(xcounts))
				ycounts = ycounts / float(np.sum(ycounts))
				if xcounts.max() > 0.66 or ycounts.max() > 0.66:
					return False

				#accept every remaining window
				return True
			else:
				# discard windows with more than 50% erroneous samples
				return False
		return False

	def getStartWindow(self, old_start, sliding_window_step_size):
		start = old_start
		starttime = self.gaze[old_start, self.ti]
		while starttime < self.gaze[old_start, self.ti] + sliding_window_step_size:
			start += 1
			if start >= len(self.gaze[:, self.xi])-1:
				return len(self.gaze[:, self.xi])-1
			starttime = self.gaze[start, self.ti]
		return start

	def getEndWindow(self, start, sliding_window_size):
		end = start
		while self.gaze[end, self.ti] < self.gaze[start, self.ti] + sliding_window_size:
			end += 1
			if end >= len(self.gaze[:, self.xi])-1:
				return len(self.gaze[:, self.xi])-1
		return end

	def get_ngram_features(self, wb):
		feat = []
		feat.append(sum(x > 0 for x in wb.values()))  # 1. size
		feat.append(np.max(wb.values()))  # 2. maximum
		nonzeros = [i for i in wb.values() if i]
		if len(nonzeros)<1:
			feat.append(0)
		else:
			feat.append(min(nonzeros))  # 3. non-zero minimum
		feat.append(np.argmax(wb.values()))  # 2. arg max
		if len(nonzeros)<1:
			feat.append(0)
		else:
			feat.append(np.argmin(np.array(nonzeros)))  # 3. arg min
		feat.append(feat[1] - feat[2])  # 4. diff max - min
		feat.append(np.mean(wb.values()))  # 5. mean of all counts
		feat.append(np.var(wb.values()))  # 6. var of all counts
		return feat
