#!/usr/bin/bash
for which_ssw in jan2019; do
    for i_gcm in 0 1 2 3 4 5 6 7 8 9 10 11; do
        python pipeline_gcms.py "$which_ssw" "$i_gcm" 
    done
done
