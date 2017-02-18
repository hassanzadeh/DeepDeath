rm ../data/models -rf
hadoop jar \
/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
-D mapreduce.job.reduces=1 \
-files hadoop \
-mapper "python hadoop/mapper.py -n 1 -r 0.05 " \
-reducer "python hadoop/reducer.py  -f 10000 " \
-input ../data/training \
-output ../data/models
