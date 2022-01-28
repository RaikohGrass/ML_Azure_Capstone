
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
```
example_data = 'https://raw.githubusercontent.com/RaikohGrass/ML_Azure_Capstone/main/precipitation_LA.csv'
dataset = Dataset.Tabular.from_delimited_files(example_data)  
```


## Automated ML
For our development we used the following configuration:
```
# TODO: Put your automl settings here
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'normalized_root_mean_squared_error'
}

# TODO: Put your automl config here
automl_config = AutoMLConfig(compute_target='EduardoCluster',
                             task = "regression",
                             training_data=dataset,
                             label_column_name="PRCP", 
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```
As optimization metric we selected the normalized root mean squared error, a common metric for regression problems.
Since we are dealing with a numeric prediction task, we selected "regression" on the "task" parameter of autoML. We enabled early stopping and used a timeout of 20 minutes for the training runs.


### Results
The resulting model was a Voting Ensemble made up of the following algorithms:
- 2'ExtremeRandomTrees',
- 2 'XGBoostRegressor'
- 'RandomForest
- 'DecisionTree

This Voting Ensemble resulted in the following metrics:

![image](https://user-images.githubusercontent.com/83981857/151494780-934f918b-427b-4b93-af4d-af606ffd7d38.png)

A way of improving the metrics would be to add more data to be used on the training or better yet to use deep learning together with the based time data that we ignored from the dataset. This way we would grant our models a feeling of the weather seasonal distributions in the year and of the consecutive days with rain.

We can see some details of the autoML run on the following screenshot:
![autoMLwidget](https://user-images.githubusercontent.com/83981857/151495233-71bf3315-079d-499f-87fd-6b468799025e.JPG)

The run and the model are shown in the following screenshot:
![autoMLrun_and_model](https://user-images.githubusercontent.com/83981857/151495385-f1e9a197-d0f8-42ef-9f0c-fb83f8bf152b.JPG)


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
