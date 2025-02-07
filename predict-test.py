#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

meteo_id = 'xyz-123'
meteo = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 24,
    "monthlycharges": 29.85,
    "totalcharges": (24 * 29.85)
}


response = requests.post(url, json=meteo).json()
print(response)

if response['churn'] == True:
    print('sending promo email to %s' % meteo_id)
else:
    print('not sending promo email to %s' % meteo_id)