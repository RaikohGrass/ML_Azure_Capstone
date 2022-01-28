
# Precipitation prediction using Microsoft Azure

The aim of this project is developing Machine Learning models to predict the precipitations in the city of Los Angeles based on meteorogical data. 
Microsoft Azure will be the platform to be used for the development and the deployment of the models. The idea is to benchmark models created using autoML and an optimized DecissionTree using Hyperdrive. The best model will be deployed and a sample data payload will be used to try out the created endpoint. 

*The dataset to be used is public and available on Kaggle.* 

## Project Set Up and Installation
The set up of our project is based on the environment provided by Udacity. No additional data science packages or add-ons are needed. A curated environment of Microsoft Azure will be enough to replicate our results

## Dataset

### Overview
As mentioned earlier the data set is made up of meteorological data from different weather stations in Los Angeles. The data includes the following features:
- Name of the station
- Time stamp of the measurement
- Average daily wind speed
- Peak gust time
- Average temperature (empty column)
- Maximum temperature
- Minimum temperature
- Direction of fastest 2-minute wind
- Precipitation <----- goal feature

The data is available in tabular form on Kaggle:
https://www.kaggle.com/varunnagpalspyz/precipitation-prediction-in-la

### Task
We will develop models using Microsoft Azure that will predict the precipitation on Los Angeles.
From the data available we will use the following features for the training of the models:

- Average daily wind speed
- Average temperature (empty column)
- Maximum temperature
- Minimum temperature
- Direction of fastest 2-minute wind

We will not include time stamps as we will not be using models with time dependant abstractions nor we will use the name of the station.

### Access
The data will be accessed through a URL from github (copy of the dataset from Kaggle):
'''
example_data = 'https://raw.githubusercontent.com/RaikohGrass/ML_Azure_Capstone/main/precipitation_LA.csv'
dataset = Dataset.Tabular.from_delimited_files(example_data)  
'''


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
