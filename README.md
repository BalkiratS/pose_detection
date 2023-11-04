# Pose Classification Project

This project focuses on training a machine learning model to classify yoga poses, specifically the "Tree Pose" and the "Warrior Pose," as either correct or incorrect based on body landmarks data obtained using BlazePose. The project involves data collection, model development, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [License](#license)

## Introduction

The project aims to classify the "Tree Pose" and "Warrior Pose" based on body landmarks data. Detailed information about the project's background, objectives, and methods can be found in the project's Jupyter Notebooks.

## Dataset

- The dataset contains body landmarks data collected using BlazePose during the execution of the "Tree Pose" and "Warrior Pose."
- The dataset is used for training, validation, and testing.
- ![Screenshot 1](Dataset/sample%Images/Screenshot.png)<!-- width="300" -->

## Model Architecture

- The project involves the development of a neural network model for pose classification.
- The model architecture includes an input layer, embedding layer, dense layers, and an output layer. Details can be found in the Jupyter Notebooks.

## Usage

- The Jupyter Notebooks (`tree.ipynb` and `warrior.ipynb`) provide code and documentation for data preprocessing, model development, training, and evaluation.
- Refer to the notebooks for a step-by-step guide on how to use the model and assess its performance.

## Repository Structure

The repository is organized as follows:

- `notebooks/`: Contains Jupyter Notebooks with code and documentation.
- `models/`: Stores the trained model files in TensorFlow and TensorFlow Lite formats.
- `data/`: Dataset and related data.
- `README.md`: This file.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.
