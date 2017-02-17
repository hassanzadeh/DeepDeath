hadoop jar \
/usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapreduce.job.reduces=5 \
-files lr\
-mapper " python lr/mapper.py -n 5 -r 0.4 " \
-reducer " python lr/reducer.py -f 10000 " \
-input cdc/training \
-output cdc/models
