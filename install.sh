#!/usr/bin/env bash

if [[ ! -z HOME_ROBOT_DIR ]]; then
    read -p "Enter the path to the home-robot root directory: " HOME_ROBOT_DIR
fi

# copy harobo to home-robot/projects if not already there
if [[ ! -d $HOME_ROBOT_DIR/projects/harobo ]]; then
    echo "Copying harobo to $HOME_ROBOT_DIR/projects"
    mkdir $HOME_ROBOT_DIR/projects/harobo
    cp -r ./* $HOME_ROBOT_DIR/projects/harobo/
    echo "Done"
else
    echo "harobo already exists in $HOME_ROBOT_DIR/projects"
fi

# copy igp model weights
if [[ ! -z HOME_ROBOT_DIR ]]; then
    echo "Copying igp model weights to $HOME_ROBOT_DIR"
    cp -r igp $HOME_ROBOT_DIR/data/checkpoints/
    echo "Done"
fi

# copy detic perception
if [[ ! -z HOME_ROBOT_DIR ]]; then
    echo "Copying detic perception to $HOME_ROBOT_DIR"
    cp -r harobo/perception/detection/detic/detic_perception_harobo.py \
         $HOME_ROBOT_DIR/src/home_robot/home_robot/perception/detection/detic/
    echo "Done"
fi