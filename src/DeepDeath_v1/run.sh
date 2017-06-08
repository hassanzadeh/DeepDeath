rm ../data/models -rf
hadoop jar \
/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
-D mapreduce.job.reduces=10 \
-D mapred.child.java.opts=-Xmx2048M \
-D mapred.tasktracker.reduce.tasks.maximum=10 \
-files hadoop \
-mapper "python hadoop/mapper.py -n 10 -r 0.2 " \
-reducer "python hadoop/reducer.py  -f 5000 " \
-input ../data/training \
-output ../data/models
