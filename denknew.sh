#!/bin/bash

 python voxel_prediction.py example_files/denk/denk.json  \
    -m train 









python voxel_prediction.py example_files/denk/denk.json  \
    -m test \
    -d /home/tbeier/src/voxel_prediction/example_files/denk/denk_raw.h5 \
    -k raw \
    --out /home/tbeier/src/voxel_prediction/example_files/denk/out/pmap.h5

