#!/bin/bash
for k in 2 4 6 8 10 12
do
echo "Approximating derivative when k = $k"
python appr_derivative.py $k o
python appr_derivative.py $k e
done
