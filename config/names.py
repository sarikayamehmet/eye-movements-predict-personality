fixations_list_labels = ['mean x', 'mean y',
                         'var x', 'var y',
                         't start', 't end',
                         'start index', 'end index',
                         'mean diameter', 'var diameter',
                         'mean successive angles', 'var successive angles'
                         ]
fix_mean_x_i = 0
fix_mean_y_i = 1
fix_var_x_i = 2
fix_var_y_i = 3
fix_start_t_i = 4
fix_end_t_i = 5
fix_start_index_i = 6
fix_end_index_i = 7
fix_mean_diam_i = 8
fix_var_diam_i = 9
fix_mean_succ_angles = 10
fix_var_succ_angles = 11

saccades_list_labels = ['start x', 'start y',
                        'end x', 'end y',
                        'angle',
                        't start', 't end',
                        'start index', 'end index',
                        'mean diameter', 'var diameter',
                        'peak velocity', 'amplitude',
                        ]

sacc_start_x_i = 0
sacc_start_y_i = 1
sacc_end_x_i = 2
sacc_end_y_i = 3
sacc_angle_i = 4
sacc_t_start_i = 5
sacc_t_end_i = 6
sacc_start_index_i = 7
sacc_end_index_i = 8
sacc_mean_diam_i = 9
sacc_var_diam_i = 10
sacc_peak_vel_i = 11
sacc_amplitude_i = 12

blink_list_labels = ['t start', 't end', 'start index', 'end index']

blink_start_t_i = 0
blink_end_ti_i = 1
blink_start_index_i = 2
blink_end_index_i = 3

event_feature_labels = ['fixation rate', 'saccade rate',  # 0 1
                        'small sacc. rate', 'large sacc. rate', 'positive sacc. rate', 'negative sacc. rate',  # 2 3 4 5
                        'ratio sacc - fix',  # 6
                        'ratio small sacc', 'ratio large sacc', 'ratio right sacc', 'ratio left sacc',  # 7 8 9 10
                        'mean sacc amplitude', 'var sacc amplitude', 'min sacc amplitude', 'max sacc amplitude',  #11 12 13 14
                        'mean peak velocity', 'var peak velocity', 'min peak velocity', 'max peak velocity',  # 15 16 17 18
                        'mean mean diameter sacc', 'var mean  diameter sacc', 'mean var diameter sacc',  # 19 20 21 22
                        'var var diameter sacc',
                        'mean fix duration', 'var fix duration', 'min fix duration', 'max fix duration',  # 23 24 25 26
                        'dwelling time',
                        'mean mean subsequent angle', 'var mean subsequent angle', 'mean var subsequent angle', 'var var subsequent angle',
                        'mean var x', 'mean var y', 'var var x', 'var var y',  # 27 28 29 30
                        'mean mean diameter fix', 'var mean diameter fix', 'mean var diameter fix', 'var var diameter fix',  # 31 32 33 34
                        'mean blink duration', 'var blink duration', 'min blink duration', 'max blink duration',  # 35 36 37 38
                        'blink rate'  # 39
                        ]

event_feature_labels_long = ['fixation rate', 'saccade rate',  # 0 1
                             'small saccade rate', 'large saccade rate', 'positive saccade rate', 'negative saccade rate',  # 2 3 4 5
                             'saccade:fixation ratio',  # 6
                             'ratio of small saccades', 'ratio of large saccades', 'ratio of right saccades', 'ratio of left saccades',  # 7 8 9 10
                             'mean saccade amplitude', 'var saccade amplitude', 'min saccade amplitude', 'max saccade amplitude',  #11 12 13 14
                             'mean saccadic peak velocity', 'var saccadic peak velocity', 'min saccadic peak velocity', 'max saccadic peak velocity',  # 15 16 17 18
                             'mean of the mean pupil diameter during saccades', 'var of the mean pupil diameter during saccades',
                             'mean of the var pupil diameter during saccades', 'var of the var pupil diameter during saccades', # 19 20 21 22
                             'mean fixation duration', 'var fixation duration', 'min fixation duration', 'max fixation duration',  # 23 24 25 26
                             'dwelling time',
                             'mean of the mean of subsequent angles', 'var of the mean of subsequent angles',
                             'mean of the var of subsequent angles', 'var of the var of subsequent angles',
                             'mean of the var of x', 'mean of the var of y', 'var of the var of x', 'var of the var of y',  # 27 28 29 30
                             'mean of the mean pupil diameter during fixations', 'var of the mean pupil diameter during fixations',
                             'mean of the var pupil diameter during fixations', 'var of the var pupil diameter during fixations',  # 31 32 33 34
                             'mean blink duration', 'var blink duration', 'min blink duration', 'max blink duration',  # 35 36 37 38
                             'blink rate'  # 39
                            ]

def get_wordbook_feature_labels(movement_abbreviation):
    return [movement_abbreviation + s + ' WB' + str(n) for n in [1, 2, 3, 4] for s in ['>0', 'max', 'min', 'arg max', 'arg min', 'range', 'mean', 'var']]

def get_wordbook_feature_labels_long(movement_abbreviation):
    return [s1 + str(n) + '-gram ' + movement_abbreviation + s2 for n in [1, 2, 3, 4]
                                                 for (s1, s2) in [('number of different ', ' movements'),
                                                                  ('max frequency ', ' movements'),
                                                                  ('min frequency ', ' movements'),
                                                                  ('most frequent ', ' movement'),
                                                                  ('least frequent ', ' movement'),
                                                                  ('range of frequencies of ', ' movements'),
                                                                  ('mean frequency of ', ' movements'),
                                                                  ('var frequency of ', ' movements')
                                                                  ]]

position_feature_labels = ['mean x', 'mean y', 'mean diameter',
                           'min x', 'min y', 'min diameter',
                           'max x', 'max y', 'max diameter',
                           'min-max x', 'min-max y', 'min-max diameter',
                           'std x', 'std y', 'std diameter',
                           'median x', 'median y', 'median diameter',
                           '1st quart x', '1st quart y', '1st quart diameter',
                           '3rd quart x', '3rd quart y', '3rd quart diameter',
                           'IQR x', 'IQR y', 'IQR diameter',
                           'mean abs diff x', 'mean abs diff y', 'mean abs diff diameter',
                           'mean diff x', 'mean diff y', 'mean diff diameter',
                           'mean subsequent angle'
                           ]

position_feature_labels_long = ['mean x', 'mean y', 'mean pupil diameter',
                                'minimum x', 'minimum y', 'minimum pupil diameter',
                                'maximum x', 'maximum y', 'maximum pupil diameter',
                                'range x', 'range y', 'range pupil diameter',
                                'std x', 'std y', 'std pupil diameter',
                                'median x', 'median y', 'median pupil diameter',
                                '1st quartile x', '1st quartile y', '1st quartile pupil diameter',
                                '3rd quartile x', '3rd quartile y', '3rd quartile pupil diameter',
                                'inter quartile range x', 'inter quartile range y', 'inter quartile range pupil diameter',
                                'mean difference of subsequent x', 'mean difference of subsequent y', 'mean difference of subsequent pupil diameters',
                                'mean diff x', 'mean diff y', 'mean diff pupil diameter',
                                'mean subsequent angle'
                                ]

heatmap_feature_labels = ['heatmap_'+str(i).zfill(2) for i in xrange(0, 64)]
heatmap_feature_labels_long = ['heatmap cell '+str(i).zfill(2) for i in xrange(0, 64)]

full_label_list = event_feature_labels + heatmap_feature_labels + position_feature_labels + \
                  get_wordbook_feature_labels('sacc.') + get_wordbook_feature_labels('SF')

full_long_label_list = event_feature_labels_long + heatmap_feature_labels_long + position_feature_labels_long + \
                  get_wordbook_feature_labels_long('sacc.') + get_wordbook_feature_labels_long('SF')


sacc_dictionary = ['A', 'B', 'C', 'R', 'E', 'F', 'G', 'D', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'U', 'u', 'b', 'r', 'f',
					'd', 'j', 'l', 'n']
sacc_bins_two = [a+b for a in sacc_dictionary for b in sacc_dictionary]
sacc_bins_three = [a+b+c for a in sacc_dictionary for b in sacc_dictionary for c in sacc_dictionary]
sacc_bins_four = [a+b+c+d for a in sacc_dictionary for b in sacc_dictionary for c in sacc_dictionary for d in sacc_dictionary]
sacc_bins = [sacc_dictionary, sacc_bins_two, sacc_bins_three, sacc_bins_four]

saccFix_dictionary = ['S_lu', 'S_ld', 'S_lr', 'S_ll', 'S_su', 'S_sd', 'S_sr', 'S_sl', 'F_l', 'F_s']
saccFix_bins_two = [a+b for a in saccFix_dictionary for b in saccFix_dictionary]
saccFix_bins_three = [a+b+c for a in saccFix_dictionary for b in saccFix_dictionary for c in saccFix_dictionary]
saccFix_bins_four = [a+b+c+d for a in saccFix_dictionary for b in saccFix_dictionary for c in saccFix_dictionary for d in saccFix_dictionary]
saccFix_bins = [saccFix_dictionary, saccFix_bins_two, saccFix_bins_three, saccFix_bins_four]

def write_pami_feature_labels_to_file(targetfile):
    f = open(targetfile, 'w')  # creates if it does not exist
    f.write(',short,long\n')
    i = 0
    for item1, item2 in zip(full_label_list, full_long_label_list):
        f.write(str(i) + ',' + item1 + ',' + item2 + '\n')
        i += 1
    f.close()
