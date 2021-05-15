
# general packages
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os

# packages for importing the cofig variables from the YAML file
import yaml

# packages for importing the variables from the command line
import argparse

# for converting strings to dates
from datetime import datetime

# packages for feature importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from statsmodels.stats.outliers_influence import variance_inflation_factor

# packages for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# package for removing outliers
from sklearn.ensemble import IsolationForest


def get_variables_from_command_line():

    """
    Read the variables in from the command line arguements

    Returns:
        (bool, bool, bool, int, int): A boolean of whether to plot the feature importance & the feature correlation
                                      A boolean of whether to print the outlier rows discovered using the isolation forest algorithm
                                      A boolean of whether to print a table of the VIF score for each feature after they are calculated
                                      An integer of how much feature importance to maintain in the features selected for the regression analysis
                                      An integer of how much feature importance to maintain in the features selected for the classification analysis
    """

    parser = argparse.ArgumentParser(description="The variables that make this program work")
    parser.add_argument(dest="plot_data", nargs="?", type=bool, default=False, help="Do you want to plot graphs of the feature importance and correlation between features?")
    parser.add_argument(dest="print_outliers", nargs="?", type=bool, default=True, help="Do you want to output the outlier rows removed in this process using the isolation forest?")
    parser.add_argument(dest="print_col_and_vif", nargs="?", type=bool, default=False, help="Do you want to print a table of the VIF scores for each column after this is calculated?")
    parser.add_argument(dest="reg_perc_importance_to_retain", nargs="?", type=int, default=0.71, help="What \% of importance do you want to maintain across the regression selected features?")
    parser.add_argument(dest="clas_perc_importance_to_retain", nargs="?", type=int, default=0.77, help="What \% of importance do you want to maintain across the classification selected features?")
    args = parser.parse_args()

    return args.plot_data, args.print_outliers, args.print_col_and_vif, args.reg_perc_importance_to_retain, args.clas_perc_importance_to_retain


def get_date_range_of_data(df):

    """
    Take in a dataframe with a date column and output the span of the data range in this dataframe

    Parameters:
        df (dataframe) : A pandas dataframe that has a columns called "date"

    Returns:
        None
    """

    min_date = min(df["date"])
    max_date = max(df["date"])
    interval_size_in_days = str(datetime.strptime(max_date, "%Y-%m-%d") - datetime.strptime(min_date, "%Y-%m-%d")).split(',')[0]

    print(" - This data conatins {} from '{}' to '{}'".format(interval_size_in_days, min_date, max_date))


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


def get_features(df, shift_period, reg_or_clas):

    if reg_or_clas == 'reg':
        # get a list of all the binary price columns needed for the classification analysis
        binary_price_cols_to_drop = list(df.filter(like="binary_price_change", axis=1).columns)

        # get a list of the shifted price columns not needed for this regression analysis
        shifted_cols = list(df.filter(like="shifted", axis=1).columns)
        train_colname = "shifted_{}".format(shift_period)
        shifted_price_cols_to_drop = [col for col in shifted_cols if col != train_colname]

    elif reg_or_clas == 'clas':
        # get a list of the shifted columns needed for the regression analysis
        shifted_price_cols_to_drop = list(df.filter(like="shifted", axis=1).columns)

        # get a list of the shifted binary price columns needed for the classification analysis
        binary_price_cols = list(df.filter(like="binary_price_change", axis=1).columns)
        train_colname = "binary_price_change_{}".format(shift_period)
        binary_price_cols_to_drop = [col for col in binary_price_cols if col != train_colname]

    cols_to_drop = ["date"] + binary_price_cols_to_drop + shifted_price_cols_to_drop

    # drop the features that are not to be used in the pca
    features = df.drop(columns=cols_to_drop)

    return features, cols_to_drop, train_colname


def get_feature_subset_using_random_forest(reg_or_clas, df, y, technical_indicator_list, period_list, rand_forest_params, random_seed):

    n_jobs = rand_forest_params["n_jobs"]
    step = rand_forest_params["step"]
    min_features_to_select = rand_forest_params["min_features_to_select"]
    verbose = rand_forest_params["verbose"]
    cv = rand_forest_params["cv"]
    scoring = rand_forest_params["scoring"]

    all_filtered_cols = []
    features_list = []
    for technichal_indicator in tqdm(technical_indicator_list):
        for period in period_list:
            
            if verbose != 0:
                print("Finding most important column for the indicator: '{}' over the period: '{}'".format(technichal_indicator, period))
            
            # look for columns that have this technichal indicator and this period
            indicator_and_period = period + technichal_indicator

            filtered_df = df.filter(like=indicator_and_period, axis=1)

            # get a list of all the columns that were analysed
            all_filtered_cols.extend(list(filtered_df.columns))

            # define a Random Forest model
            if reg_or_clas == 'reg':
                rf_model = RandomForestRegressor(random_state=random_seed, n_jobs=n_jobs)
            elif reg_or_clas == 'clas':
                rf_model = RandomForestClassifier(random_state=random_seed, n_jobs=n_jobs)

            # Rank the features using recursive feature elimination and cross-validated selection of the best number of features
            rfecv = RFECV(rf_model, step=step, min_features_to_select=min_features_to_select, verbose=verbose, cv=cv, scoring=scoring, n_jobs=n_jobs)
            #print(filtered_df[filtered_df.isna().any(axis=1)])
            #big_cols = list(filtered_df.columns[(filtered_df > 2147483647).any().to_list()])
            #na_cols = list(filtered_df.columns[filtered_df.isna().any().to_list()])
            #print(big_cols, na_cols)
            #inval = filtered_df[big_cols + na_cols]
            #print(inval[(inval > 2147483647)].dropna())
            #print(inval.dtypes)
            rfecv.fit(filtered_df, y)
            
            # if more than one feature was identifies as the best feature
            if rfecv.n_features_ > 1:
                rf_model.fit(filtered_df, y)
                maximp = rf_model.feature_importances_.max()
                
                for i in range(len(rf_model.feature_importances_)):
                    if maximp == rf_model.feature_importances_[i]:
                        new_features = filtered_df.columns[i]
            
            # if only one feature was identified as the best feature
            else:
                mask = rfecv.get_support()
                new_features = filtered_df.columns[mask][0]
            
            # add this feature to a list        
            features_list.append(str(new_features))

            if verbose != 0:
                print(" -", indicator_and_period + ':', new_features, "\n")

    # sorted the list of features
    features_list.sort()

    return features_list, all_filtered_cols


def drop_high_vif(df, thresh, print_dropped_vif, df_name):
    
    # get a list of the indexes for all the columns
    variables = list(range(len(df.columns)))
    
    # change the datatypes to numeric values
    df = df.apply(pd.to_numeric, errors='coerce')
        
    dropped_cols = []
    dropped = True
    i = 1
    while dropped:
        dropped = False
        
        # subset the df to the not dropped variables
        relevant_cols = df.iloc[:, variables]

        # calculate the variance inflation factor for each column
        vif = [variance_inflation_factor(relevant_cols.values, ix) for ix in range(relevant_cols.shape[1])]
        
        # if the maximum vif is above our threshold, drop the column that this corresponds to
        if max(vif) > thresh:
            # find the column with the maximum VIF and drop it
            maxloc = vif.index(max(vif))
            dropped_cols.append(relevant_cols.columns[maxloc])
            del variables[maxloc]
            dropped = True

        print(i, end=" ")
        i += 1

    print()

    # create a map of the columns to their VIF value
    map_col_to_vif = {"column": df.columns[variables],
                      "vif": vif
                     }
    col_to_vif_df = pd.DataFrame(map_col_to_vif)
    
    if print_dropped_vif:
        # print the variables that weren't dropped
        if len(variables) == len(df.columns):
            print("All columns in the {} dataframe still remaining, none were dropped".format(df_name))
        
        else:
            print("---{}---".format(df_name))
            print("Dropped:\n", dropped_cols)
            print('Remaining variables:\n', list(df.columns[variables]))
        
    return df.iloc[:, variables], col_to_vif_df
    

def remove_multicollinearity_using_VIF(reg_or_clas, indicator_df, no_indicators_df, print_vif_outputs):

    # remove mulitcollinearity for the indicators
    dropped_indicator_df, indicator_col_to_vif_df = drop_high_vif(indicator_df, thresh=5, print_dropped_vif=print_vif_outputs, df_name="Indicators")

    # remove mulitcollinearity for the non-indicators
    dropped_non_indicator_df, non_indicator_col_to_vif_df = drop_high_vif(no_indicators_df, thresh=10, print_dropped_vif=print_vif_outputs, df_name="Non Indicators")

    # Add the indicator & non indicator filtered subset of columns together into one table
    merged_df = pd.concat([dropped_non_indicator_df, dropped_indicator_df], axis=1)

    # remove mulitcollinearity for the merged dataframe
    if reg_or_clas == 'reg':
        dropped_merged_df, merged_col_to_vif_df = drop_high_vif(merged_df, thresh=10, print_dropped_vif=print_vif_outputs, df_name="Merged")
    elif reg_or_clas == 'clas':
        dropped_merged_df, merged_col_to_vif_df = drop_high_vif(merged_df, thresh=5, print_dropped_vif=print_vif_outputs, df_name="Merged")

    if print_vif_outputs:
        print("----------------\nIndicators\n----------------")
        print(indicator_col_to_vif_df.to_markdown())
        print("\n----------------\nNon-Indicators\n----------------")
        print(non_indicator_col_to_vif_df.to_markdown())
        print("\n----------------\nMerged Indicators & Non-Indicators\n----------------")
        print(merged_col_to_vif_df.to_markdown())

    return dropped_merged_df


def use_rand_forests_and_vif_to_subset_features(reg_or_clas, train_features, target_array, config_variables, train_colname, print_vif_outputs):

    # extract the variables from the config file
    technical_indicator_list = config_variables["technical_indicator_list"]
    technichal_indicator_time_period_list = config_variables["technichal_indicator_time_period_list"]
    rand_forest_params = config_variables["rand_forest_feat_select_params"]
    random_seed = config_variables["random_seed"]

    # Use a Random Forest model to get the most important column for each technichal indicator & period combo
    indicators_features_list, all_filtered_cols = get_feature_subset_using_random_forest(reg_or_clas, train_features, target_array, technical_indicator_list, technichal_indicator_time_period_list, rand_forest_params, random_seed)
    reduced_indicator_features_df = train_features[indicators_features_list]

    # Use the Variance Inflation Factor (VIF) to get rid of multicollinearity between columns
    no_indicators_df = train_features.drop(columns=all_filtered_cols + [train_colname], axis=1)
    print(len(indicators_features_list), len(no_indicators_df.columns))
    dropped_merged_df = remove_multicollinearity_using_VIF(reg_or_clas, reduced_indicator_features_df, no_indicators_df, print_vif_outputs)

    return dropped_merged_df


def use_rand_forest_to_get_feature_importance(reg_or_clas, feature_df, y, random_seed):

    # fit the RFC model
    if reg_or_clas == 'reg':
        rf_final = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
    if reg_or_clas == 'clas':
        rf_final = RandomForestClassifier(random_state=random_seed, n_jobs=-1)
    rf_final.fit(feature_df, y)

    # make a dataframe of the feature importance
    feature_importance_dict = {"feature": feature_df.columns, "importance": rf_final.feature_importances_}
    feature_importance_df = pd.DataFrame(feature_importance_dict).sort_values(["importance"]).reset_index(drop=True)

    return feature_importance_df


def maintain_important_features_up_to_specified_percent(feature_importance_df, percent_importance_to_retain):

    """
    Take in a dataframe of features and their importances and get a list of the top 'n' features that accumulatively have a specified amount of feature importance

    Parameters:
        feature_importance_df (dataframe) : a pandas dataframe containing features of the data and their calculated importances
        percent_importance_to_retain (int) : the amount of accumulative feature importance that is to be maintained from the selected features

    Returns:
        (list) : a list of the features required to get the specified accumulative amount of feature importance
    """

    # get the top 'N' features that give me the specified % of importance retained
    important_features = []
    i = total_importance = 0
    while total_importance < percent_importance_to_retain:
        row = feature_importance_df.loc[i, ]
        
        important_features.append(row["feature"])
        total_importance += row["importance"]
        i = i + 1
        
    print("The top {} features maintain {}\% of the feature importance".format(len(important_features), percent_importance_to_retain*100))

    return important_features


def plot_feature_importance(feature_importance_df):

    """
    Plot a bar chart of the feature importance for each feature in the input dataframe

    Parameters:
        feature_importance_df (dataframe) : a pandas dataframe with two columns, a column of features and a column of their importance

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(feature_importance_df["feature"], feature_importance_df["importance"], align='center')
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    plt.show()


def create_correlation_plot(features_df):

    """
    Plot a correlation plot of all the featurs in the input dataframe

    Parameters:
        features_df (dataframe) : a pandas dataframe containing features as its columns

    Returns:
        None
    """

    corr = features_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(26, 13))

    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f')


def further_subset_features_using_feat_importance(reg_or_clas, dropped_merged_df, target_array, percent_importance_to_retain, plot_data, random_seed):

    # Use a Random Forest Classifier to select the most important features
    feature_importance_df = use_rand_forest_to_get_feature_importance(reg_or_clas, dropped_merged_df, target_array, random_seed)
    
    # subset the main dataframe to specified selected columns based on the % importance they retain
    important_features = maintain_important_features_up_to_specified_percent(feature_importance_df, percent_importance_to_retain)
    selected_features_df = dropped_merged_df[important_features].sort_index(axis=1)

    if plot_data:
        # plot the feature importance & the correlation between features
        plot_feature_importance(feature_importance_df)
        create_correlation_plot(selected_features_df)

    return selected_features_df


def get_outlier_rows_using_IF(y, reg_or_clas):

    target_price_array = np.reshape(y, (-1,1))

    # Define the outlier model
    if reg_or_clas == 'reg':
        outlier = IsolationForest(contamination=0.9)
    elif reg_or_clas == 'clas':
        outlier = IsolationForest(contamination=0)

    # fit the model to the target data
    outlier.fit(target_price_array)

    # deduce if there are any outliers in the data
    outs = outlier.predict(target_price_array)

    # create a dataframe stating if the row is an outlier or not where '1' means it is not outlier and '-1' means it is an outlier
    outlier_df = pd.DataFrame(zip(outs), columns=["outlier"])

    # Find the outlier rows
    outlier_indexes = list(outlier_df.query('outlier == -1').index)

    return outlier_indexes


def remove_outliers(df, target_array, reg_or_clas, print_outliers):

    # Get a list of the outlier rows using an Isolation Forest algorithm
    outlier_indexes = get_outlier_rows_using_IF(target_array, reg_or_clas)
    if print_outliers and not df.iloc[outlier_indexes, ].empty:
        print("The rows with indexes {} are outliers".format(outlier_indexes))
        print(df.iloc[outlier_indexes, ].to_markdown())
        
    # drop the rows where there was an outlier
    final_df = df.drop(outlier_indexes, axis=0).reset_index(drop=True)

    return final_df


def main():

    # read in the variables from the YAML configuration file
    with open("../../Config_files/config.yaml", "r") as variables:
        config_variables = yaml.load(variables, Loader=yaml.FullLoader)
    data_directory = config_variables["data_directory"]
    types_of_analysis = config_variables["types_of_analysis"]
    shift_periods_in_days_list = config_variables["shift_periods_in_days_list"]
    train_intervals_list = config_variables["train_intervals_list"]
    random_seed = config_variables["random_seed"]

    # read in the processed data
    processed_data = pd.read_csv(os.path.join(data_directory, 'processed_data.csv'))

    # define the variables from the command line
    plot_data, print_outliers, print_col_and_vif, reg_percent_importance_to_retain, clas_percent_importance_to_retain = get_variables_from_command_line()

    # Iterate through the intervals, # features and shift periods & select the most important features from the data
    for reg_or_clas in tqdm(types_of_analysis):
        print("==========\n  ", reg_or_clas, "  \n==========")
        
        # set the percentage of feature importance to retain
        percent_importance_to_retain = reg_percent_importance_to_retain if reg_or_clas == 'reg' else clas_percent_importance_to_retain
        
        for interval_num in range(1, len(train_intervals_list)+1):
            print("-------------")
            print("Interval:", interval_num)

            # get the start and end dates for this interval
            train_start_date, train_end_date = train_intervals_list[interval_num-1]

            # define the train data for this date interval
            train_df = fs.get_subset_of_data(processed_data, train_start_date, train_end_date)
            fs.get_date_range_of_data(train_df) 

            # iterate over the shift periods and select the features based on this period
            for shift_period in shift_periods_in_days_list:

                # get a dataframe of the data's features
                train_features, train_cols_to_drop, train_colname = fs.get_features(train_df, shift_period, reg_or_clas)
            
                # get a numpy array of the feature we are trying to predict
                y = np.ravel(train_features[train_colname])

                # use Random Forest model and Variance Inflation Factor (VIF) to get the most important columns
                dropped_merged_df = fs.use_rand_forests_and_vif_to_subset_features(reg_or_clas, train_features, y, config_variables, train_colname, print_vif_outputs)

                if 'price' not in dropped_merged_df.columns:
                    dropped_merged_df['price'] = train_features['price']

                # Use random forest to score feature importance - then only keep specified % importance
                selected_features_df = fs.further_subset_features_using_feat_importance(reg_or_clas, dropped_merged_df, y, percent_importance_to_retain, plot_data, random_seed)

                # add the created price column back into the data
                selected_features_df["_".join(train_colname.split('_')[:-1])] = y

                # Remove the outliers in the data using an Isolation forest
                final_df = fs.remove_outliers(selected_features_df, y, reg_or_clas, print_outliers)

                # Output the data to a CSV
                train_path = os.path.join(data_directory, 'Feat_select', '{}_rf_vif_and_if_interval_{}_shift_{}.csv'.format(reg_or_clas, interval_num, shift_period))
                final_df.to_csv(train_path, index=False)

if __name__ == '__main__':
    main()