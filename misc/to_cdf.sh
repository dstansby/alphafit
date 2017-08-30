#!/bin/bash
for i in $(find . -name '*.csv')
do
  echo "$i"
  Qtran -oc -h/home/dstansby/corefit/misc/helios_corefit.qfh $i
  echo "$i done!"
done
