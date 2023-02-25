#!/usr/bin/bash

src_data=hdfs://hobot-bigdata/user/binbin.zhang/data/aishell/data
data=data

echo "Download prepared data from HDFS"
hdfs dfs -get $src_data $data
