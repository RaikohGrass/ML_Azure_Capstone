from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.tree import DecisionTreeRegressor

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
path ="https://raw.githubusercontent.com/RaikohGrass/ML_Azure_Capstone/main/precipitation_LA.csv"

ds = TabularDatasetFactory.from_delimited_files(path)

def clean_data(data):
    
    
    df = ds.to_pandas_dataframe()
    # We drop the empty column TAVG
    df = df.drop('TAVG')
    # Since we are not using deep learning or techniques that have a notion of time we can drop the date and PGTM columns
    # We also drop the station name column
    df = df.drop(['STATION','NAME','DATE','PGTM'])
    x = df
    y = df.pop('PRCP')

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = clean_data(ds)


run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--criterion', type=str, default='squared_error', help="Function to measure the quality of a split")
    parser.add_argument('--max_depth', type=int, default=None, help="Maximum depth for the trees")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum number of samples needed for a split")
    parser.add_argument('--max_leaf_nodes', type=int, default=None, help="Maximum number of nodes")


    args = parser.parse_args()

    run.log("Split Criterion:", np.str(args.criterion))
    run.log("Max depth:", np.int(args.max_depth))
    run.log("Min samples split:", np.int(args.min_samples_split))
    run.log("Max leaf nodes:", np.int(args.max_leaf_nodes))

    model = LogisticRegression(criterion = args.criterion,
                               max_depth = args.max_depth,
                               min_samples_split = args.min_samples_split,
                               max_leaf_nodes = args.max_leaf_nodes).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    #Save the model
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()