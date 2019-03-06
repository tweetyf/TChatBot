#!/bin/sh
basepath=$(cd `dirname $0`; pwd)
cd $basepath
worker1='nohup python3 '${basepath}'/server.py 127.0.0.1 10240 >logs 2>&1 &'
echo $worker1
$worker1 &
exit 0

