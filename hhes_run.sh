#!/bin/bash
#python voxel_prediction.py example_files/hhes/hhes.json  \
#    -m train 

# tiny roi
#python voxel_prediction.py example_files/hhes/hhes.json  \
#    -m test -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_3.h5 -k data --out hhes_3_n_out.h5 -rb 100 100 100 -re 300 300 200

# 
#python voxel_prediction.py example_files/hhes/hhes.json  -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_1.h5 -k data \
#    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n.h5
#
#python voxel_prediction.py example_files/hhes/hhes.json  -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_2.h5 -k data \
#    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_2.h5
#    
#python voxel_prediction.py example_files/hhes/hhes.json  -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_3.h5 -k data \
#    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_3.h5
#
#python voxel_prediction.py example_files/hhes/hhes.json  -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_4.h5 -k data \
#    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_4.h5
#
#python voxel_prediction.py example_files/hhes/hhes.json  -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_5.h5 -k data \
#    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_5.h5  
#
#python voxel_prediction.py example_files/hhes/hhes.json  -m test \
#    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_6.h5 -k data \
#    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_6.h5  

python voxel_prediction.py example_files/hhes/hhes.json  -m test \
    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_7.h5 -k data \
    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_7.h5  

python voxel_prediction.py example_files/hhes/hhes.json  -m test \
    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_8.h5 -k data \
    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_8.h5  

python voxel_prediction.py example_files/hhes/hhes.json  -m test \
    -d /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/data_sub_n_9.h5 -k data \
    --out /media/tbeier/data/datasets/hhess/2x2x2nm_chunked/predictions/semantic/p_data_sub_n_9.h5  