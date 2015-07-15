#!/bin/bash

python voxel_prediction.py example_files/hhes_r2/hhes_r2.json  \
    -m train 






# tiny roi
#python voxel_prediction.py example_files/hhes_r2/hhes_r2.json  \
#    -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/prediction_and_raw/semantic/pr_data_sub_n_3.h5 \
#    -k data --out Fhhes_r2_3_n_out.h5 -rb 100 100 100 -re 500 500 200 




nice -n 10 python voxel_prediction.py example_files/hhes_r2/hhes_r2.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic/pr_data_sub_n_3.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/semantic_r2/p_data_sub_n_3.h5


nice -n 10 python voxel_prediction.py example_files/hhes_r2/hhes_r2.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic/pr_data_sub_n_4.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/semantic_r2/p_data_sub_n_4.h5


nice -n 10 python voxel_prediction.py example_files/hhes_r2/hhes_r2.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic/pr_data_sub_n_5.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/semantic_r2/p_data_sub_n_5.h5


nice -n 10 python voxel_prediction.py example_files/hhes_r2/hhes_r2.json  \
    -m test \
    -d /home/tbeier/hhes_r2/prediction_and_raw/semantic/pr_data_sub_n_6.h5 \
    -k data \
    --out /home/tbeier/hhes_r2/predictions/semantic_r2/p_data_sub_n_6.h5