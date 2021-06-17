#!/bin/bash

rclone copy -P gdrive:Research/Datasets/Raw/mer-taffc/ /storage/s3/raw/mer-taffc/
rclone copy -P gdrive:Research/Datasets/Splits/mer-taffc-kfold/ /storage/s3/splits/mer-taffc-kfold/

rclone copy -P gdrive:Research/Datasets/Raw/deam/ /storage/s3/raw/deam/
rclone copy -P gdrive:Research/Datasets/Splits/deam-kfold/ /storage/s3/splits/deam-kfold/

rclone copy -P gdrive:Research/Datasets/Raw/emomusic/ /storage/s3/raw/emomusic/
rclone copy -P gdrive:Research/Datasets/Splits/emomusic-kfold/ /storage/s3/splits/emomusic-kfold/

rclone copy -P gdrive:Research/Datasets/Raw/pmemo/ /storage/s3/raw/pmemo/
rclone copy -P gdrive:Research/Datasets/Splits/pmemo-kfold/ /storage/s3/splits/pmemo-kfold/