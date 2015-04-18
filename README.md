#
# Running on andrew unix
#
# 1. Make sure spark is installed and set environment variable:

export PATH=/afs/cs.cmu.edu/project/bigML/spark-1.3.0-bin-hadoop2.4/bin:$PATH

# 2. Submit job using spark-submit:

spark-submit --master local dsgd_mf.py \
<num_factors> <num_workers> <num_iterations> <beta> <lambda> <input_V_path> <output_W_path> <output_H_path>

#
# Running on amazon emr
#
# 1. Create EMR cluster with spark installed with the following bootstrap action 

s3://support.elasticmapreduce/spark/install-spark

# 2. SSH into EMR master and Set environment variable:

export PATH=/home/hadoop/spark/bin:$PATH

# 3. Upload data to HDFS

# 4. Submit job using spark-submit:
spark-submit --master yarn-client dsgd_mf.py \
<num_factors> <num_workers> <num_iterations> <beta> <lambda> <input_V_path> <output_W_path> <output_H_path>


