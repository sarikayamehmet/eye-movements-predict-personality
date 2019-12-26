# eye-movements-predict-personality
Eye Movements During Everyday Behavior Predict Personality Traits (code for https://doi.org/10.3389/fnhum.2018.00105)

# Eye movements during everyday behavior predict personality traits
*Sabrina Hoppe, Tobias Loetscher, Stephanie Morey and Andreas Bulling*

This repository provides all data and code used for the publication [in Frontiers in Human Neuroscience](https://dx.doi.org/10.3389/fnhum.2018.00105).  

## Dataset
   *  Gaze data recorded at 60Hz from 42 participants is stored in `data/ParticipantXX`.  
      For each participant there are three files:
      1.  `events.csv` is a list of gaze events as provided by the SMI eye tracker software.
          The list contains saccades, fixations and blinks but only the blink information was used in the code.
      2.  `gaze_positions.csv` is a table with three columns: time in seconds, x gaze coordinate and y gaze coordinate. The x and y coordinates describe the participants' gaze direction normalised to the range from 0 to 1.
      3.  `pupil_diameter.csv` is another table with three columns: time in seconds, diameter of the right eye and diameter of the left eye. The diameter values are absolute gaze estimates in mm.  

   All files are of the same length and each row corresponds to one data sample. That is, the n-th row in all three files belongs to the same point in time.

   * Ground truth personality scores from the respective questionnaires, participant age and sex (1: male, 2: female) can be found in `info/personality_sex_age.csv`.  

   * Personality score ranges that were obtained by binning the questionnaire scores are provided in `info/binned_personality.csv`.  

   * Timestamps indicating the times when participants entered and left the shop are given in `info/annotation.csv` in seconds.  


## Code
reproducing the paper results step by step:
1. __Extract features from raw gaze data__:    
   `python 00_compute_features.py` to compute gaze features for all participants  
   Once extracted, the features are stored in `features/ParticipantXX/window_features_YY.npy` where XX is the participant number and YY the length of the sliding window in seconds.  


2. __Train random forest classifiers__  
   `./01 train_classifiers.sh` to reproduce the evaluation setting described in the paper in which each classifier was trained 100 times.  
  `./02_train_specialized_classifiers.sh` to train specialized classifiers on parts of the data (specifically on data from inside the shop or on the way).

   If the scripts cannot be executed, you might not have the right access permissions to do so. On Linux, you can try `chmod +x 01_train_classifiers.sh`,`chmod +x 02_train_specialized_classifiers.sh` and `chmod +x 03_label_permutation_test.sh` (see below for when/how to use the last script).

   In case you want to call the script differently, e.g. to speed-up the computation or try with different parameters, you can pass the following arguments to `classifiers.train_classifier`:  
    `-t` 	trait index between 0 and 6  
    `-l`   lowest number of repetitions, e.g. 0   
    `-m`   max number of repetitions, e.g. 100  
    `-a`   using partial data only: 0 (all data), 1 (way data), 2(shop data)  

   In case of performance issues, it might be useful to check `_conf.py` and change `max_n_jobs` to restrict the number of jobs (i.e. threads) running in parallel.

   The results will be saved in `results/A0` for all data, `results/A1` for way data only and `results/A2` for data inside a shop. Each file is named `TTT_XXX.npz`, where TTT is the abbreviation of the personality trait (`O`,`C`,`E`,`A`,`N` for the Big Five and `CEI` or `PCS` for the two curiosity measures). XXX enumerates the classifiers (remember that we always train 100 classifiers for evaluation because there is some randomness involved in the training process).  

3. __Train baselines__
   * To train a classifier that always predicts the most frequent personality score range from its current training set, please execute `python 03_train_baseline.py`  
   * To train classifiers on permuted labels, i.e. perform the so-called label permutation test, please execute `./04_label_permutation_test.sh`    


4. __Performance analysis__
   * Run `python 05_plot_weights.py` to extract feature importance scores. These scores will be visualized in `figures/figure2.pdf` which corresponds to Figure 2 in the paper and `figures/table2.tex` which is shown in Table 2 in the supplementary information.
   (additionally this step computes F1 scores which are required for the next step, so do not skip it)
   * The results obtained from both baselines will be written to disk and read once you execute `python 06_baselines.py`.
   A figure illustrating the actual classifiers' performance along with the random results will be written to `figures/figure1.pdf` as well as `figures/figure1.csv` and correspond to Figure 1 in the paper.  


5. __Context comparison__  
`python 07_evaluation_across_contexts.py` to compute the average correlation coefficients between predictions based on data from different contexts. The table with all coefficients will be written to `figures/table1-5.csv` which can be found in Table 1 and Table 5 in supplementary information.  
If (some) files in the results folder are missing, try re-running all one of the bash (\*.sh) scripts again.

6. __Descriptive analysis__  
`python 08_descriptive.py` to compute the correlation between each participant's average feature for the most frequently chosen time window and their personality score range. Results are written to four files `figures/table4-1.tex`,`figures/table4-2.tex`,`figures/table4-3.tex`,`figures/table4-4.tex` and are shown together in Table 4 in the supplementary information.  

7. __Window Size Histogram__    
`python 09_plot_ws_hist.py` to plot a histogram of window sizes chosen during the nested cross validation routine to `figures/ws_hist.pdf`.

All these scripts write intermediate results to disk, i.e. if you start a script a second time, it will be much faster - but the first run can take some time, e.g. up to 8 hours to train classifiers for one context on a 16 core machine; 1 hour to compute correlations between contexts.  

## Citation  
If you want to cite this project, please use the following Bibtex format:

```
@article{hoppe18_fhns,
title = {Eye Movements During Everyday Behavior Predict Personality Traits},
author = {Sabrina Hoppe and Tobias Loetscher and Stephanie Morey and Andreas Bulling},
url = {https://perceptual.mpi-inf.mpg.de/files/2018/04/hoppe18_fhns.pdf
https://github.molgen.mpg.de/sabrina-hoppe/everyday-eye-movements-predict-personality
https://www.newscientist.com/article/2167850-ai-can-predict-your-personality-just-by-how-your-eyes-move/
http://www.dailymail.co.uk/sciencetech/article-5686817/An-incredible-mind-reading-AI-predict-personality-just-studying-eyes-move.html
https://www.digitaltrends.com/cool-tech/ai-personality-eye-movement/
http://www.newsweek.com/artificial-intelligence-algorithm-can-work-out-your-personality-simply-909752
https://www.12news.com/video/syndication/veuer/new-ai-can-predict-personality-from-eye-movements/602-8116328
https://www.usatoday.com/videos/tech/news/2018/05/03/new-ai-can-predict-personality-eye-movements/34526179/},
doi = {10.3389/fnhum.2018.00105},
year = {2018},
date = {2018-03-05},
journal = {Frontiers in Human Neuroscience},
volume = {12},
pages = {105:1-105:8},
abstract = {Besides allowing us to perceive our surroundings, eye movements are also a window into our mind and a rich source of information on who we are, how we feel, and what we do. Here we show that eye movements during an everyday task predict aspects of our personality. We tracked eye movements of 42 participants while they ran an errand on a university campus and subsequently assessed their personality traits using well-established questionnaires. Using a state-of-the-art machine learning method and a rich set of features encoding different eye movement characteristics, we were able to reliably predict four of the Big Five personality traits (neuroticism, extraversion, agreeableness, conscientiousness) as well as perceptual curiosity only from eye movements. Further analysis revealed new relations between previously neglected eye movement characteristics and personality. Our findings demonstrate a considerable influence of personality on everyday eye movement control, thereby complementing earlier studies in laboratory settings. Improving automatic recognition and interpretation of human social signals is an important endeavor, enabling innovative design of human–computer systems capable of sensing spontaneous natural user behavior to facilitate efficient interaction and personalization.},
keywords = {},
pubstate = {published},
tppubtype = {article}
}
```
