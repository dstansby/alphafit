#!/bin/bash

HEADER="/Users/dstansby/github/corefit/misc/helios_corefit.qfh"
# This should be run from the root of directories in which the processed csv
# files reside
for i in $(find . -name '*.csv')
do
  echo "$i"
  ./Qtran -oc -f -h$HEADER $i
  echo "$i done!"
done
