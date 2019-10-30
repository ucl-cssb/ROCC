#!/bin/sh


for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
  python ../fitted_Q_on_double_aux.py -s ./perterbed_system_no_reset -r $i
done
