n_parallel_jobs=6 # number of jobs that will be run in parallel
n_total_jobs=100
for li in $(seq 0 $n_parallel_jobs $n_total_jobs);
do
	# the for loop below will start n_parallel_jobs for each trait
	# unless n_total_jobs will be reached first (this is checked in the if statementc)
	mi=`expr "$li" + "$n_parallel_jobs"`
	if [ "$mi" -gt "$n_total_jobs" ]
	then
		mi=$n_total_jobs
	fi

	for t in $(seq 0 6);
	do
		python2.7 -m classifiers.train_classifier -t $t -s 0 -a 0 -l $li -m $mi
		wait
	done
done
