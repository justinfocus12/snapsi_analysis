#!/bin/bash
for f in logs/*."$1"; do 
    echo $f $(cat $f | wc -w)
done
