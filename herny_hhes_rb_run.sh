#!/bin/bash

python voxel_prediction.py example_files/hhes_rb/hhes_rb.json  \
    -m train 


#tiny roi
python voxel_prediction.py example_files/hhes_rb/hhes_rb.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic_r2/pr_data_sub_n_5.h5 \
    -k data --out roi_rb_5_n_out.h5 -rb 100 100 100 -re 500 500 200 



nice -n 10 python voxel_prediction.py example_files/hhes_rb/hhes_rb.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic_r2/pr_data_sub_n_5.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/binary/p_data_sub_n_5.h5


nice -n 10 python voxel_prediction.py example_files/hhes_rb/hhes_rb.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic_r2/pr_data_sub_n_7.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/binary/p_data_sub_n_7.h5


nice -n 10 python voxel_prediction.py example_files/hhes_rb/hhes_rb.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic_r2/pr_data_sub_n_8.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/binary/p_data_sub_n_8.h5