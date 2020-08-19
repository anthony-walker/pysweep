#!/bin/bash
echo $LIMIT1 $LIMIT2
for id in $(seq $LIMIT1 1 $LIMIT2)
do
    scancel $id
done