# Customer conversion rate prediction

Welcome to the **Customer conversion rate prediction** project! This repository provides tools and resources for predicting customer conversion rate based on synthetic dataset in order to help marketing teams in business decision.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Setup Instructions](#setup-instructions)

## Overview

This project helps businesses identify potentiel customer with high conversion probability and enabling more effective marketing strategies cibling.

## Dataset

The dataset used for this project is from Kaggle dataset. The dataset  "Customer Conversion Prediction Dataset" is a synthetic dataset created for the purpose of simulating a customer conversion prediction scenario in order to help stackholder, business decision maker and marketing teams. The dataset contains customer's perfornel information such as age, gender, location and informations about customer interaction with a hypothetical business or website such as lead source, lead status, ect. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-conversion-dataset-for-stuffmart-com/data)

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
   docker run -d -p 5000:5000 predict-app
   
