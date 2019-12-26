import numpy as np
import matplotlib.pyplot as plt
import seaborn
from config import conf
import os

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

hist_sum_sum = np.sum(hist_sum)

plt.figure()
ax = plt.subplot(111)
bars = ax.bar(conf.all_window_sizes, hist_sum/float(hist_sum_sum)*100, width=8, tick_label=[str(x) for x in conf.all_window_sizes])

for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % (height/100.*hist_sum_sum),
                ha='center', va='bottom')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('window size in s')
plt.ylabel('percentage')
plt.savefig('figures/ws_hist.pdf')
plt.close()
