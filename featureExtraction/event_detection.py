import numpy as np
import sys
import math
from config import names as gs


def get_fixation_list(gaze, errors, xi, yi, ti, fixation_radius_threshold, fixation_duration_threshold, pupil_diameter):
	n, m = gaze.shape
	fixations = []
	fixation = []  # single fixation, to be appended to fixations
	counter = 0  # number of points in the fixation
	sumx = 0 # used to compute the center of a fixation in x and y direction
	sumy = 0
	distance = 0 # captures the distance of a current sample from the fixation center
	i = 0 # iterates through the gaze samples

	while i < n - 1:
		x = gaze[i, xi]
		y = gaze[i, yi]

		if counter == 0:
			# ignore erroneous samples before a fixation
			if errors[i]:
				i += 1
				continue
			centerx = x
			centery = y
		else:
			centerx = np.true_divide(sumx, counter)
			centery = np.true_divide(sumy, counter)

		if not errors[i]: # only update distance if the current sample is not erroneous
			distance = np.sqrt((x - centerx) * (x - centerx) + (y - centery) * (y - centery))

		if distance > fixation_radius_threshold:  # start new fixation
			if gaze[(i - 1), ti] - gaze[(i - counter), ti] >= fixation_duration_threshold:
				start_index = i - counter + 1
				end_index = i - 1 - 1

				# discard fixations with more than 50% erroneous samples
				percentage_error = np.sum(errors[start_index:(end_index + 1)]) / float(end_index - start_index)
				if percentage_error >= 0.5:
					if errors[i]:
						i += 1
						counter = 0
					else:
						counter = 1
						sumx = x
						sumy = y
					continue

				gaze_indices = np.arange(start_index, end_index+1)[np.logical_not(errors[start_index:(end_index + 1)])]

				start_index = gaze_indices[0]
				end_index = gaze_indices[-1]

				gazex = gaze[start_index:(end_index + 1), xi][np.logical_not(errors[start_index:(end_index + 1)])]
				gazey = gaze[start_index:(end_index + 1), yi][np.logical_not(errors[start_index:(end_index + 1)])]
				gazet = gaze[start_index:(end_index + 1), ti][np.logical_not(errors[start_index:(end_index + 1)])]

				# extract fixation characteristics
				fixation.append(np.mean(gazex))  # 0.-1. mean x,y
				fixation.append(np.mean(gazey))
				fixation.append(np.var(gazex))  # 2-3. var x, y
				fixation.append(np.var(gazey))
				fixation.append(gazet[0])  # 4-5. t_start, t_end
				fixation.append(gazet[-1])
				fixation.append(gaze_indices[0])  # 6-7. index_start, index_end
				fixation.append(gaze_indices[-1])

				ds = ((pupil_diameter[start_index:(end_index+1), 1] + pupil_diameter[start_index:(end_index+1), 2]) / 2.)[np.logical_not(errors[start_index:(end_index+1)])]

				fixation.append(np.mean(ds))  # 8. mean pupil diameter
				fixation.append(np.var(ds))  # 9. var pupil diameter

				succ_dx = gazex[1:] - gazex[:-1]
				succ_dy = gazey[1:] - gazey[:-1]
				succ_angles = np.arctan2(succ_dy, succ_dx)

				fixation.append(np.mean(succ_angles))  # 10 mean successive angle
				fixation.append(np.var(succ_angles))  # 11 var successive angle
				fixations.append(fixation)
				assert len(fixation) == len(gs.fixations_list_labels)

			# set up new fixation
			fixation = []
			if errors[i]:
				i += 1
				counter = 0
			else:
				counter = 1
				sumx = x
				sumy = y
		else:
			if not errors[i]:
				counter += 1
				sumx += x
				sumy += y

		i += 1
	return fixations


def get_saccade_list(gaze, fixations, xi, yi, ti, pupil_diameter, fixation_radius_threshold, errors,
						saccade_min_velocity, max_saccade_duration):
	saccades = []
	wordbook_string = []

	# each movement between two subsequent fixations could be a saccade, but
	for i in xrange(1, len(fixations)):
		# ...not if the window is too long
		duration = float(fixations[i][gs.fix_start_t_i] - fixations[i - 1][gs.fix_end_t_i])
		if duration > max_saccade_duration:
			continue

		start_index = fixations[i - 1][gs.fix_end_index_i]
		end_index = fixations[i][gs.fix_start_index_i]

		gazex = gaze[start_index:(end_index + 1), xi][np.logical_not(errors[start_index:(end_index + 1)])]
		gazey = gaze[start_index:(end_index + 1), yi][np.logical_not(errors[start_index:(end_index + 1)])]
		gazet = gaze[start_index:(end_index + 1), ti][np.logical_not(errors[start_index:(end_index + 1)])]

		dx = np.abs(gazex[1:] - gazex[:-1])
		dy = np.abs(gazey[1:] - gazey[:-1])
		dt = np.abs(gazet[1:] - gazet[:-1])

		# ...not if less than 2 non-errouneous amples are left:
		if len(dt) < 2:
			continue

		distance = np.linalg.norm([dx, dy])
		peak_velocity = np.amax(distance / dt)

		start_x = gazex[0]
		start_y = gazey[0]
		end_x = gazex[-1]
		end_y = gazey[-1]

		dx = end_x - start_x
		dy = end_y - start_y

		# ...not if the amplitude is shorter than a fith of fixation_radius_threshold
		amplitude = np.linalg.norm([dx, dy])
		if amplitude < fixation_radius_threshold / 5.0:
			continue

		# ...not if the peak velocity is very low
		if peak_velocity < saccade_min_velocity:
			continue


		percentage_error = np.sum(errors[start_index:(end_index + 1)]) / float(end_index - start_index)
		# ...not if more than 50% of the data are erroneous
		if percentage_error >= 0.5:
			continue
		else:  # found saccade!
			# compute characteristics of the saccade, like start and end point, amplitude, ...
			saccade = []
			saccade.append(start_x)  # 0.-1. start x,y
			saccade.append(start_y)
			saccade.append(end_x)  # 2-3. end x,y
			saccade.append(end_y)

			if dx == 0:
				radians = 0
			else:
				radians = np.arctan(np.true_divide(dy, dx))

			if dx > 0:
				if dy < 0:
					radians += (2 * np.pi)
			else:
				radians = np.pi + radians

			saccade.append(radians)  # 4. angle
			saccade.append(fixations[i - 1][gs.fix_end_t_i])  # 5-6. t_start, t_end
			saccade.append(fixations[i][gs.fix_start_t_i])
			saccade.append(start_index)  # 7-8. index_start, index_end
			saccade.append(end_index)

			ds = (pupil_diameter[start_index:(end_index + 1), 1] + pupil_diameter[start_index:(end_index + 1),
			                                                       2]) / 2.0
			saccade.append(np.mean(ds))  # 9. mean pupil diameter
			saccade.append(np.var(ds))  # 10. var pupil diameter
			saccade.append(peak_velocity)  # 11. peak velocity

			saccade.append(amplitude)  # 12. amplitude

			# append character representing this kind of saccade to the wordbook_string which will be used for n-gram features
			sac_id = get_dictionary_entry_for_saccade(amplitude, fixation_radius_threshold, radians)
			wordbook_string.append(sac_id)
			saccades.append(saccade)

			# assert all saccade characteristics were computed
			assert len(saccade) == len(gs.saccades_list_labels)
	return saccades, wordbook_string


def get_blink_list(event_strings, gaze, ti):
	assert len(event_strings) == len(gaze)

	# detect Blinks
	blinks = []
	blink = []  # single blink, to be appended to blinks
	i = 0
	starti = i
	blink_started = False

	while i < len(event_strings) - 1:
		if event_strings[i] == 'Blink' and not blink_started:  # start new blink
			starti = i
			blink_started = True
		elif blink_started and not event_strings[i] == 'Blink':
			blink.append(gaze[starti, ti])
			blink.append(gaze[i - 1, ti])
			blink.append(starti)
			blink.append(i - 1)
			blinks.append(blink)
			assert len(blink) == len(gs.blink_list_labels)
			blink_started = False
			blink = []
		i += 1

	return blinks


def get_dictionary_entry_for_saccade(amplitude, fixation_radius_threshold, degree_radians):
	# Saacade Type: small, iff amplitude less than 2 fixation_radius_thresholds
	#						U
	#					O		A
	#				N		u		B
	#			M		n		b		C
	#		L		l				r		R
	#			K		j		f		E
	#				J		d		F
	#					H		G
	#						D


	degrees = np.true_divide(degree_radians * 180.0, np.pi)
	if amplitude < 2 * fixation_radius_threshold:
		d_degrees = degrees / (np.true_divide(90, 4))
		if d_degrees < 1:
			sac_id = 'r'
		elif d_degrees < 3:
			sac_id = 'b'
		elif d_degrees < 5:
			sac_id = 'u'
		elif d_degrees < 7:
			sac_id = 'n'
		elif d_degrees < 9:
			sac_id = 'l'
		elif d_degrees < 11:
			sac_id = 'j'
		elif d_degrees < 13:
			sac_id = 'd'
		elif d_degrees < 15:
			sac_id = 'f'
		elif d_degrees < 16:
			sac_id = 'r'
		else:
			print ('error! d_degrees cannot be matched to a sac_id for a small saccade ', d_degrees)
			sys.exit(1)

	else:  # large
		d_degrees = degrees / (np.true_divide(90, 8))

		if d_degrees < 1:
			sac_id = 'R'
		elif d_degrees < 3:
			sac_id = 'C'
		elif d_degrees < 5:
			sac_id = 'B'
		elif d_degrees < 7:
			sac_id = 'A'
		elif d_degrees < 9:
			sac_id = 'U'
		elif d_degrees < 11:
			sac_id = 'O'
		elif d_degrees < 13:
			sac_id = 'N'
		elif d_degrees < 15:
			sac_id = 'M'
		elif d_degrees < 17:
			sac_id = 'L'
		elif d_degrees < 19:
			sac_id = 'K'
		elif d_degrees < 21:
			sac_id = 'J'
		elif d_degrees < 23:
			sac_id = 'H'
		elif d_degrees < 25:
			sac_id = 'D'
		elif d_degrees < 27:
			sac_id = 'G'
		elif d_degrees < 29:
			sac_id = 'F'
		elif d_degrees < 31:
			sac_id = 'E'
		elif d_degrees < 33:
			sac_id = 'R'
		else:
			print ('error! d_degrees cannot be matched to a sac_id for a large saccade: ', d_degrees)
			sys.exit(1)
	return sac_id


def detect_all(gaze, errors, ti, xi, yi, fixation_radius_threshold=0.01, pupil_diameter=None, event_strings=None,
               fixation_duration_threshold=0.1, saccade_min_velocity=2, max_saccade_duration=0.1):
	"""
	:param gaze: gaze data, typically [t,x,y]
	:param fixation_radius_threshold: dispersion threshold
	:param fixation_duration_threshold: temporal threshold
	:param ti, xi, yi: index data for gaze,i.e. for [t,x,y] ti=0, xi=1, yi=2
	:param pupil_diameter: pupil diameters values, same length as gaze
	:param event_strings: list of events, here provided by SMI. used to extract blink information
	"""

	fixations = get_fixation_list(gaze, errors, xi, yi, ti, fixation_radius_threshold, fixation_duration_threshold,
		                              pupil_diameter)
	saccades, wordbook_string = get_saccade_list(gaze, fixations, xi, yi, ti, pupil_diameter,
		                                             fixation_radius_threshold, errors, saccade_min_velocity,
		                                             max_saccade_duration)
	blinks = get_blink_list(event_strings, gaze, ti)

	return fixations, saccades, blinks, wordbook_string
