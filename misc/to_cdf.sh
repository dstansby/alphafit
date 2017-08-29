#!/bin/bash
for i in $(find . -name '*.csv')
do
  echo "$i"
  Qtran -oc –h helios_corefit.qfh $i
  echo "$i done!"
  exit
done
