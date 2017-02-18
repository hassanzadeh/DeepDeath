rm ../data/models -rf
hadoop jar \
/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
-D mapreduce.job.reduces=4 \
-D mapred.child.java.opts=-Xmx2048M \
-D mapred.tasktracker.map.tasks.maximum \
-files hadoop \
-mapper "python hadoop/mapper.py -n 10 -r 0.2 " \
-reducer "python hadoop/reducer.py  -f 10000 " \
-input ../data/training \
-output ../data/models
