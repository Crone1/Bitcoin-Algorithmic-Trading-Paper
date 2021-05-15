

# general packages
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

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
# |            CALCULATE DATE RANGE              |
# ================================================

def check_if_theres_a_row_for_every_day(df, min_date, max_date, interval_size_in_days):

    dates_with_error = []
    for days_since_start, (_, row) in zip(range(interval_size_in_days), df.iterrows()):
        if min_date + timedelta(days=days_since_start) != row["date"].date():

            print(min_date + timedelta(days=days_since_start), row["date"].date())

            dates_with_error.append((min_date + timedelta(days=days_since_start), row["date"].date()))

    return dates_with_error


def get_date_range_of_data(df):

    """
    Take in a dataframe with a date column and output the span of the data range in this dataframe

    Parameters:
        df (dataframe) : A pandas dataframe that has a columns called "date"

    Returns:
        None
    """

    min_date = datetime.strptime(min(df["date"]), '%Y-%m-%d').date()
    max_date = datetime.strptime(max(df["date"]), '%Y-%m-%d').date()
    interval_size_in_days = int(str(max_date - min_date).split(",")[0].split(" ")[0])

    print("Scraped {} days of data - from '{}' to '{}'".format(interval_size_in_days, min_date, max_date))

    if interval_size_in_days != len(df) - 1:
        dates_with_error = check_if_theres_a_row_for_every_day(df, min_date, max_date, interval_size_in_days)
        print("The following dates had a problem in them:")
        print(dates_with_error)



# ================================================
# |                 CHECK NANS                   |
# ================================================

def is_nan(val):
    """
    Check if a value is 'Nan'

    Params:
        val: any type - can be a variable of any datatype

    Return:
        Boolean: Whether the value is an 'Nan' value or not
    """
    return val != val


def get_cols_with_nan(df):

    nan_cols = []
    for col in df.columns:
        any_nan = df[col].isnull().values.any()
        if any_nan:
            nan_cols.append(col)

    return nan_cols


def find_col_nan_ranges(df, output=True):

    # get the columns that contain an 'nan' value
    nan_cols = get_cols_with_nan(df)

    col_to_nan_dates = {}
    for col in tqdm(nan_cols):
        prev_was_nan = False
        for index, row in df.iterrows():
            if is_nan(row[col]) and not prev_was_nan:
                prev_was_nan = True
                if "date" in row:
                    nan_start_date = str(row["date"])
                else:
                    nan_start_date = str(index)

            elif not is_nan(row[col]) and prev_was_nan:
                prev_was_nan = False
                if "date" in row:
                    nan_end_date = str(row["date"])
                else:
                    nan_end_date = str(index)

                if col in col_to_nan_dates:
                    col_to_nan_dates[col].append((nan_start_date, nan_end_date))
                else:
                    col_to_nan_dates[col] = [(nan_start_date, nan_end_date)]
                    
    # print these columns
    if output:
        print("---------------------------------------------------------------------")
        print("{} columns had a 'NaN' value in them:".format(len(nan_cols)))
        pprint(nan_cols)
        print("---------------------------------------------------------------------")
        print("The date ranges in these columns where the NaN's are located are:")
        pprint(col_to_nan_dates)

    return col_to_nan_dates


def find_the_latest_date_each_column_starts(nan_col_dates, n_cols_to_print):

    # store when each column starts (stops being NaN)
    nan_date_list = []
    nan_col_list = []
    for col, date_list in nan_col_dates.items():
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
    cols_printed = 0
    for d, c in zip(sorted_dates[::-1], sorted_cols[::-1]):
        if cols_printed == n_cols_to_print:
            break
        print(c, "-->", d)
        cols_printed += 1



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
    month = df["date"].apply(lambda date: date.split("-")[1])
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
        last_month = m
        new_month_list.append(new_m)

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