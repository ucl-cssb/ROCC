#!/bin/sh


for i in 41 42 43 44 45 46 47 48 49
do
  python ../reviewer_experiments.py -s ./reviewer_flipped -r $i
done
