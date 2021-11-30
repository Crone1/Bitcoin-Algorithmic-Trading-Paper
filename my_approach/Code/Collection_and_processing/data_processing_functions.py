
# general packages
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import os

# for converting strings to dates
from datetime import datetime

# packages for processing data
from sklearn.impute import SimpleImputer

# packages for printing dictionaries nicely
from pprint import pprint

# pakcages for cyclical time features
import math

# packages for the correlation plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# ================================================
# |         READ IN THE INTERNAL DATA            |
# ================================================

def read_raw_internal_data_into_list(data_directory, config_variables):

    # unpack the variables needed for this scrape from the configuration dictionary
    bitcoinity_features_list = config_variables["bitcoinity_features_list"]

    list_of_dfs = []
    for file_name in [f for f in os.listdir(data_directory) if "raw_internal_data" in f]:
        raw_data = pd.read_csv(os.path.join(data_directory, file_name))
        
        # for the bitcoinity data, merge the data values from the various exchanges into one column for each feature
        df = merge_different_bitcoinity_exchanges_into_one_col(raw_data, bitcoinity_features_list) if "bitcoinity" in file_name else raw_data

        list_of_dfs.append(df)

    return list_of_dfs


def merge_different_bitcoinity_exchanges_into_one_col(df, list_of_features):

    # set up the dataframe with the merged features as columns
    condensed_features_df = pd.DataFrame(columns=["date"] + list_of_features)

    # add the date column to this dataframe
    condensed_features_df["date"] = df["date"]

    # iterate over the other columns and add these too the dataframe
    merged_cols = ["date"]
    for feature in list_of_features:
        # get a list of all the columns associated with the specified feature
        different_exchange_cols = list(df.filter(like=feature, axis=1).columns)
        merged_cols.extend(different_exchange_cols)

        if feature != 'rank':
            # average the values in this column to form one column
            condensed_features_df[feature] = df[different_exchange_cols].mean(axis=1)
        else:
            condensed_features_df.drop(columns=[feature], inplace=True)
            condensed_features_df[different_exchange_cols] = df[different_exchange_cols]

    # check if there were any columns in the dataframe that werent covered by the list of features
    cols_not_in_feat_list = []
    for col in df.columns:
        if col not in merged_cols:
            cols_not_in_feat_list.append(col)
    if cols_not_in_feat_list:
        print("The following columns were dropped from the dataframe during this step as they didn't have an associated feature:")
        print(cols_not_in_feat_list)

    return condensed_features_df


def merge_dfs_on_col(list_of_dfs, col):

    # merge the dataframes on the specified column
    merged_df = list_of_dfs[0]
    for df in list_of_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=col, how="outer")

    # return the dataframe
    return merged_df


def merge_internal_dfs(list_of_dfs, config_variables):

    # Merge the dataframes
    merged_df = merge_dfs_on_col(list_of_dfs, "date")
        
    # sort the df by the date
    sorted_merged_df = merged_df.sort_values("date").reset_index(drop=True)

    # turn the same feature columns from both these data sources into one column
    cols_to_join = config_variables["bitcoin_internal_data_cols_to_join"]
    for col_name, (c1, c2) in cols_to_join.items():
        sorted_merged_df[col_name] = sorted_merged_df[[c1, c2]].mean(axis=1)
        sorted_merged_df.drop(columns=[c1, c2], inplace=True)

    return sorted_merged_df



# ================================================
# |             READ DATA INTO LIST              |
# ================================================

def read_data_into_list(data_directory, name_in_filename):

    list_of_dfs = []
    for file_name in [f for f in os.listdir(data_directory) if name_in_filename in f]:
        df = pd.read_csv(os.path.join(data_directory, file_name))
        list_of_dfs.append(df)

    return list_of_dfs



# ================================================
# |             FILL MISSING VALUES              |
# ================================================

def fill_in_missing_values(df):

    # fill the 'NaN' volume values with '0'
    volume_cols = [c for c in df.columns if ("volume" in c)]
    volume_df = df.loc[:, volume_cols]
    filled_volume_df = volume_df.fillna(value=0, axis=0)

    # forward fill the 'NaN' values in the other columns
    other_cols = [c for c in df.columns if ("volume" not in c)]
    other_df = df.loc[:, other_cols]
    filled_other_df = other_df.fillna(method='ffill', axis=0)

    # join these two dataframes back together into one filled df
    filled_df = pd.concat([filled_volume_df, filled_other_df], axis=1)

    # put the columns in this filled df back in the original order
    ordered_cols_filled_df = filled_df[list(df.columns)]

    # return the dataframe
    return ordered_cols_filled_df



# ================================================
# |                 CHECK NANS                   |
# ================================================

def find_the_latest_date_each_column_starts(nan_col_dates, n_cols_to_print):

    # check if there are no nan cols
    if not nan_col_dates:
        print("There are no NaN columns in this dataset")
        return

    # store when each column starts (stops being NaN)
    nan_date_list = []
    nan_col_list = []
    for col, date_list in nan_col_dates.items():
        # Ensure all NaN values except those at the start have been filled
        if len(date_list) > 1:
            print(col)
            print(date_list)
        else:
            finish_date = date_list[0][1]
            nan_date_list.append(finish_date)
            nan_col_list.append(col)

    # sort these lists on the date
    sorted_dates, sorted_cols = zip(*sorted(zip(nan_date_list, nan_col_list)))

    # output these columns to see what are the latest dates that the columns start
    max_lenth_col = len(max(sorted_cols, key=len))
    for i, (d, c) in enumerate(zip(sorted_dates[::-1], sorted_cols[::-1])):
        if i == n_cols_to_print:
            break
        print(c.ljust(max_lenth_col), "-->", d)



# ================================================
# |             MERGING DATA SOURCES             |
# ================================================


def read_in_all_data_and_merge_it(data_directory, list_of_files, start_date):
    
    # iterate through each file
    merged_df = pd.DataFrame()
    for file in list_of_files:

        # read in the data
        filepath = os.path.join(data_directory, file)
        df = pd.read_csv(filepath)

        # set the date to its index
        df = df.set_index("date")

        # add this data to a dataframe of all the data
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how="outer")

    # return the dataframe with the date as its own columns
    return merged_df.reset_index()



# ================================================
# |            CREATE CYCLICAL TIME              |
# ================================================

def create_cyclical_day_features(df):

    # create a list of integers, one for each day (row) in the dataframe
    num_for_each_day = np.linspace(0, len(df)-1, len(df), dtype=int)

    # divide this by 7 to specify what day of the week it is
    dotw = pd.DataFrame(num_for_each_day).apply(lambda n: n%7)

    # normalize the values to match with the 0-2π cycle
    normalised_dotw = (2 * math.pi * dotw) / dotw.max()

    # return the two features that specify the cyclical nature of the day of the week
    return np.cos(normalised_dotw), np.sin(normalised_dotw)


def create_cyclical_month_features(df):

    # get the month that each day is in
    month = df["date"].apply(lambda date: str(date).split("-")[1])
    seen_months = {}
    new_month_list = []
    last_month = 0
    for m in month:
        m = int(m)
        # this is a new month, update the dict
        if m != last_month:
            if m not in seen_months:
                seen_months[m] = 0
            else:
                seen_months[m] += 1

        # save the value for this month
        new_m = m + (12 * seen_months[m])
        new_month_list.append(new_m)
        last_month = m

    # create a list of integers, one for each month in the dataframe
    num_for_each_month = np.array(new_month_list)

    # divide this by 12 to specify what month of the year it is
    dotm = pd.DataFrame(num_for_each_month).apply(lambda n: n%12)

    # normalize the values to match with the 0-2π cycle
    normalised_dotm = (2 * math.pi * dotm) / dotm.max()

    # return the two features that specify the cyclical nature of the day of the month
    return np.cos(normalised_dotm), np.sin(normalised_dotm)



# ================================================
# |               ADD SHIFT COLS                 |
# ================================================

def create_shifted_price_column(input_df, shift_period):

    # copy the dataframe so that we don't overwrite it
    df = input_df.copy()

    # get the name of the shifted column
    shifted_col = 'shifted_{}'.format(shift_period)
    
    # Shift the price column by the desired number of periods & fill the new NAN values introduced by the shift with '1'

    shifted_price = df['price'].shift(-shift_period, fill_value=1)
    
    # Add this shifted price to the dataframe
    shifted_price_col = pd.DataFrame(shifted_price).rename(columns={'price':shifted_col})
    
    return shifted_price_col


def create_shifted_binary_price_change_column(input_df, shifted_period):

    # copy the dataframe so that we don't overwrite it
    df = input_df.copy()
    
    # get the name of the shifted columns
    shifted_col = 'shifted_{}'.format(shifted_period)
    binary_col_name = "binary_price_change_{}".format(shifted_period)
    
    # Obtain the difference between the price and the shifted price to see if there was an increase/decrease over this period
    df['difference'] = df[shifted_col] - df['price']
    
    # Make a binary column of the difference and add this to the dataframe
    binary_change = [1 if val >= 0 else 0 for val in df['difference']]
    
    return pd.DataFrame(binary_change, columns=[binary_col_name])


def add_all_shifted_price_cols(df, periods_in_days):

	shifted_price_df = df.copy()
	for period in periods_in_days:
	    
	    # add shifted price col
	    price_shift_df = create_shifted_price_column(shifted_price_df, period)
	    shifted_price_df = pd.concat([shifted_price_df, price_shift_df], axis=1)

	    # add binary shifted price column
	    binary_shift_df = create_shifted_binary_price_change_column(shifted_price_df, period)
	    shifted_price_df = pd.concat([shifted_price_df, binary_shift_df], axis=1)

	return shifted_price_df


# ================================================
# |               CORRELATION MAP                |
# ================================================

def create_correlation_plot(features_df):

    """
    Plot a correlation plot of all the featurs in the input dataframe

    Parameters:
        features_df (dataframe) : a pandas dataframe containing features as its columns

    Returns:
        None
    """
    
    corr = features_df.astype(float).corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    
    # apply mask to correlation
    correlation_mask = np.where(mask == 0, corr, 0)

    fig, ax =  plt.subplots(figsize=(50, 50))
    ax.imshow(correlation_mask, cmap='coolwarm')

    # Set up the matplotlib figure
    #f, ax = plt.subplots()#figsize=(26, 13))
    #print("3. Figure defined")
    
    #sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f')


def get_feature_correlation_table(correlation_matrix):
    
    # use the correlation table to get a sorted list of the most correlated feature pairs
    full_sorted_correlation_df = pd.DataFrame(correlation_matrix.unstack().sort_values(kind="quicksort", ascending=True)).reset_index()

    # rename the columns
    full_sorted_correlation_df.columns = ["feat_1", "feat_2", "correlation"]

    # remove feature correlations where the correlation value is NaN
    no_nan_df = full_sorted_correlation_df[full_sorted_correlation_df["correlation"].notna()]

    # remove feature correlations with itself
    removed_self_corr = no_nan_df[no_nan_df["feat_1"] != no_nan_df["feat_2"]]

    # create new column with the sorted values of the two features
    removed_self_corr['sorted_feat'] = [str(sorted([a,b])) for a,b in zip(removed_self_corr["feat_1"], removed_self_corr["feat_2"])]
    
    # drop columns with duplicate values for this sorted features column
    unique_df = removed_self_corr.drop_duplicates(subset=['sorted_feat']).reset_index(drop=True)
    
    # return the df with the added column removed
    return unique_df.drop(columns=["sorted_feat"])


def count_corr_features_over_threshold(correlation_df, threshold=0.99):

    # get the correlated rows over the threshold
    neg_df = correlation_df[correlation_df["correlation"] < -threshold]
    pos_df = correlation_df[correlation_df["correlation"] > threshold].iloc[::-1].reset_index(drop=True)

    # count these rows
    print("{} feature pairs had a negative correlation over -{}".format(len(neg_df), threshold))
    print("{} feature pairs had a positive correlation over {}".format(len(pos_df), threshold))

    # return these df rows over the threshold
    return neg_df, pos_df

