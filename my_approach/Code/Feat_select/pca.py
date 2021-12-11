
# general packages
import pandas as pd
import numpy as np
import os

# packages for importing the variables from the config file
import yaml

# for converting strings to dates
from datetime import datetime

# packages for scaling the data
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# pakages for the PCA
from sklearn.decomposition import PCA
from joblib import dump

# packages for plotting
import matplotlib.pyplot as plt

    

def get_subset_of_data(df, start_date, end_date):
    
    """
    Take in a dataframe with a date column and output the subset of this dataframe using the specified date range of the input dataframe

    Parameters:
        df (dataframe) : A pandas dataframe that has a columns called "date"
        start_date (string) : A String of the minimum date we want in the subset dataframe
        end_date (string) : A String of the maximum date we want in the subset dataframe

    Returns:
        (dataframe) : A pandas dataframe which is a subset of the original dataframe whose data only contains values in between the specified start and end date
    """

    return df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop=True)


def get_features(train_df):

    """
    Remove the target columns from the input dataframe - this includes the shifted price columns used for regression and the shifted binary price change columns used for the classification

    Parameters:
        train_df (dataframe) : A pandas dataframe that has a columns called "date"

    Returns:
        (dataframe, list) : A pandas dataframe which is a subset of the input dataframe with no date or target columns & then a list of the columns dropped from the input dataframe
    """

    # get a list of the shifted columns needed for the regression analysis
    shifted_cols = list(train_df.filter(like="shifted", axis=1).columns)

    # get a list of the shifted binary price columns needed for the classification analysis
    binary_price_cols = list(train_df.filter(like="binary_price_change", axis=1).columns)

    cols_to_drop = ["date"] + shifted_cols + binary_price_cols

    # drop the features that are not to be used in the pca
    train_features = train_df.drop(columns=cols_to_drop)

    return train_features, cols_to_drop


def get_cols_with_nan(df):

    nan_cols = []
    for col in df.columns:
        if df[col].isnull().values.any():
            nan_cols.append(col)

    return nan_cols


def drop_nan_cols(df):

    nan_cols = get_cols_with_nan(df)
    print("\nDropping the following 'NaN' columns:\n", nan_cols, "\n")

    dropped = df.copy()
    return dropped.drop(columns=nan_cols)


def scale_the_data(estimators_list, x_train, verbose):

    """
    Scale the input data using the input estimators

    Parameters:
        estimators_list (list) : a list of the scaling estimators to use to scale the data
        x_train (np.array) : a numpy array containing the training features

    Returns:
        (np.array) : a numpy array of the scaled training features
    """

    # define the pipeline
    scale = Pipeline(estimators_list, verbose=verbose)
    scale.fit(x_train)

    # scale the data
    scaled_x_train = scale.transform(x_train)

    return scaled_x_train


def train_pca_model(x_train, num_features, random_seed):

    """
    Create a PCA model with a specified number of components using the input data

    Parameters:
        num_features (int) : The number of features to use in the PCA model
        random_seed (int) : The random seed to use when creating the PCA model
        x_train (np.array) : a numpy array containing the training features to train the PCA model on and to transform to the principal components

    Returns:
        (PCA model) : the trained PCA model
    """

    # get the 'n' most important principal components
    pca = PCA(n_components=num_features, random_state=random_seed)
    pca.fit(x_train)

    return pca


def plot_explained_variance(x_train, random_seed, num_features, max_variance_interval=2.5):

    """
    Plot the explained variance of each PCA component calculated using the input dataset

    Parameters:
        x_train (np.array) : a numpy array containing the training features to train the PCA model on and to transform to the principal components
        random_seed (int) : The random seed to use when creating the PCA model
        num_features (int) : The number of features to use in the PCA model
        max_variance_interval (float) : 

    Returns:
        None
    """

    # get the 'n' most important principal components
    pca = train_pca_model(x_train, num_features, random_seed)

    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative variance (%)')
    plt.title('Explained Variance')
    plt.show()

    # define a dataframe of the explained variances for the components
    expl_var_df = pd.DataFrame(columns=["component_num", "explained_variance", "cumulative_explained_variance"])
    cumulative = 0
    for comp_num, var in enumerate(list(pca.explained_variance_ratio_)):
        cumulative += var
        component_var_df = pd.DataFrame({"component_num": comp_num, "explained_variance": var, "cumulative_explained_variance": cumulative}, index=[0])
        expl_var_df = pd.concat([expl_var_df, component_var_df]).reset_index(drop=True)

    # subset this dataframe
    keep_indexes = []
    for index, row in expl_var_df.iterrows():

        if index != 0:
            div_curr = (row["cumulative_explained_variance"]*100) // max_variance_interval
            div_prev = (expl_var_df.loc[index-1, "cumulative_explained_variance"]*100) // max_variance_interval

        if (index % 5 == 0) or (div_curr > div_prev):
            keep_indexes.append(index)

    print(expl_var_df.loc[keep_indexes, :].to_markdown())


def main():

    # read in YAML configuration file
    with open("../../Config_files/config.yaml", "r") as variables:
        config_variables = yaml.load(variables, Loader=yaml.FullLoader)

    random_seed = config_variables["random_seed"]
    num_pca_features_list = config_variables["num_pca_features_list"]
    data_directory = config_variables["data_directory"]
    model_directory = config_variables["model_directory"]
    train_intervals_list = config_variables["train_intervals_list"]

    # read in the data
    processed_data = pd.read_csv(os.path.join(data_directory, 'Processed_data', 'full_processed_data.csv'))

    # define the estimator to scale the data
    estimator = [['minmax', MinMaxScaler(feature_range=(-1, 1))],
                ]

    # Iterate through the intervals and plot their explained variance
    interval_to_data = {}
    for train_start_date, train_end_date in tqdm(train_intervals_list):
        print("-------------\nTime Interval:", train_start_date, "-->", train_end_date)
        time_interval_str = '{}_to_{}'.format(train_start_date, train_end_date)

        # define the train data for this date interval
        train_df = pca.get_subset_of_data(processed_data, train_start_date, train_end_date)

        # drop any features that have Nan's in them
        no_nan_train_df = pca.drop_nan_cols(train_df)
        
        # get a dataframe of the data's features
        train_features, dropped_cols = pca.get_features(no_nan_train_df)

        # scale these features
        scaled_x_train = pca.scale_the_data(estimator, train_features, verbose=False)

        # plot the explained variance
        pca.plot_explained_variance(scaled_x_train, random_seed, num_features=100, max_variance_interval=2.5)
        
        # store the scaled features and the dropped target prices for this interval
        interval_to_data[time_interval_str] = (scaled_x_train, train_df[dropped_cols])

    # Perform the Principal Component Analysis (PCA)
    for (time_interval_str, (scaled_x_train, target_cols_df)) in tqdm(interval_to_data.items()):

        # iterate through the number of features to train the model on 
        for num_features in num_pca_features_list:

            # train PCA model
            pca_model = pca.train_pca_model(scaled_x_train, num_features, random_seed)

            # export the model for tranfroming other datasets
            pca_model_path = os.path.join(model_directory, 'PCA', 'pca_{}__{}.joblib'.format(num_features, time_interval_str))
            dump(pca_model, pca_model_path)

            # transform the train features to the PCA models 'n' principal components
            pca_train_features = pd.DataFrame(pca_model.transform(scaled_x_train))

            # add the dropped shifted and binary price change columns back into the data
            for col in target_cols_df.columns:
                pca_train_features[col] = target_cols_df[col]

            # output these table as a dataframe
            train_path = os.path.join(data_directory, 'Feat_select', 'pca_{}__{}.csv'.format(num_features, time_interval_str))
            pca_train_features.to_csv(train_path, index=False)


if __name__ == '__main__':
    main()
