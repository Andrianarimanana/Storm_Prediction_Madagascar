# Storm prediction in north of Madagascar

Welcome to the **Storm prediction in north of Madagascar** project! This repository provides tools and resources for predicting thunderstorms in northern Madagascar based on Meteosat Second Generation Data on Cloud-Top Temperature in order to climate risk management.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Setup Instructions](#setup-instructions)

## Overview

The porpuse of this project is a machine learning focused on forcasting thunderstorms in northern Madagascar, particularly around Nosy Be. The project aims to provide accurate short-term predictions (0–6 hours) to mitigate risks, protect lives, and support emergency responses in this vulnerable region.

## Dataset

The repository includes data derived from [EUMETSAT](https://www.eumetsat.int/)'s Meteosat Second Generation satellite observations , focusing on cloud-top temperature. Storms are detected in real-time using the 2D-wavelet transform method by [Klein et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017JD027432), with database creation contributed by [Rakotomanga et al. (2024)](https://eps.leeds.ac.uk/maths/pgr/13569/mendrika-rakotomanga).

Dataset Details:

Inputs: Each row includes observation time (year, month, day, hour, minute), geographic coordinates (latitude and longitude), and storm characteristics (intensity, size, and distance).
Labels: Training data includes binary labels for storm occurrence predictions at 1-hour (Storm_NosyBe_1h) and 3-hour (Storm_NosyBe_3h) lead times.
Time Span: Data spans November to April from 2004–2019 for training and 2020–2024 for testing.
Files in Repository:

train.csv: Contains input features and labels for model training.
test.csv: Contains input features without labels for model evaluation. 

## Setup Instructions

To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo-name.git

2. Activate virtual environment (make sure pipenv is already installed):
   ```bash
   pipenv shell

3. Install Dependencies:
   ```bash
   pipenv install

4. Activate the Virtual Environment
   ```bash
   pipenv shell

5. Run the project locally with pipenv
    ```bash
   # train the model
   pypenv python train.py

   # do prediction
   pipenv run python predict.py

To set up this projet using Docker Container

1. Build the docker image (make sure docker is already installed):
   ```bash
   docker build -t predict-app .

2. Running the docker container:
   ```bash
      docker run -it --rm -p 9696:9696 predict-app
   
