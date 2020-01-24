#!/bin/sh


for i in 2 3
do
  python ../traditional_neural_fitted_Q.py -s ./PID_comp -r $i
done
