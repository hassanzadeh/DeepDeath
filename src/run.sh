hadoop jar \
/usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
-D mapreduce.job.reduces=1 \
-files hadoop \
-mapper "/opt/conda/bin/python hadoop/mapper.py -n 1 -r 0.01 " \
-reducer "/opt/conda/bin/python hadoop/reducer.py  -f 10000 " \
-input /cdc/training \
-output /cdc/models
