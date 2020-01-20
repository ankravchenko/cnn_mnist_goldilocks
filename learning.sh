#!/bin/bash
printf '\n' > goldilocks_logfile.log
for i in {1..6}
do
	for j in 1 2 3 4 5 6
	do
		a=$(($i*5000))
		b=$(($j*5000))
		python3 goldilocks_output.py $a $b digit1 oddeven50 add 
		printf '	'>> goldilocks_logfile.log
	done
	printf '\n' >> goldilocks_logfile.log
	echo "finished for"$a" "$b
done
cp goldilocks_logfile.log goldilocks_logfile_backup.log
