#!/bin/bash

path=${1:-"help"}
usage="usage: ./incremental_sfm.sh path"

if [[ "$path" =~ help|--help|-h ]]
then
    echo $usage
    exit
else
    echo "Starting incremental SfM..."
    echo "This may take a while..."
    cd $path
    ls db_images/ | grep server > new-images-list.txt
    ls db_images/ | grep client >> new-images-list.txt
    ls db_images/ | grep add >> new-images-list.txt
    mkdir sparse/extra
    colmap feature_extractor --database_path database.db --image_path db_images/ --image_list_path new-images-list.txt --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.max_image_size 2400
    colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1
    colmap image_registrator --database_path database.db --input_path sparse/0 --output_path sparse/extra/
    colmap model_converter --input_path sparse/extra/ --output_path sparse/extra/ --output_type TXT
    cat sparse/extra/images.txt | grep server > gt_poses.txt
    cat sparse/extra/images.txt | grep client >> gt_poses.txt
    cat sparse/extra/images.txt | grep add >> gt_poses.txt
    echo "Done. You may find gt_poses.txt in $path"
fi
