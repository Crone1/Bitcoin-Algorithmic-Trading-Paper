
# general packages
import pandas as pd
from tqdm.auto import tqdm
import time
import os
import sys
from pprint import pprint
import numpy as np

# packages for getting bitcoin internal data - bitcoinity
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By

# packages for getting bitcoin internal data - bitinfocharts
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# for converting strings to dates
from datetime import datetime, timedelta

# packages for getting stock data
import yfinance as yf

# packages for getting quandl economic data
import quandl
from quandl.errors.quandl_error import NotFoundError

# packages for getting tweet data
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import twint



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

    min_date = min(df["date"]).date()
    max_date = max(df["date"]).date()
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
                    nan_start_date = str(row["date"].date())
                else:
                    nan_start_date = str(index)

            elif not is_nan(row[col]) and prev_was_nan:
                prev_was_nan = False
                if "date" in row:
                    nan_end_date = str(row["date"].date())
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



# ================================================
# |               BITCOINITY.ORG                 |
# ================================================

def merge_different_exchanges_into_one_col(df, list_of_features):

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


def read_exported_csvs_into_df(num_csvs, metric_names, downloads_folder):
    
    """
    Read in the csv's exported to the downlaods folder when scraping the data
    Merge the data from all these csv's into one table
    Return this table
    """
    
    # read in the first data file
    file_path = os.path.join(downloads_folder, 'bitcoinity_data.csv')
    all_data_df = pd.read_csv(file_path)
    
    # fix the column names
    name = metric_names[0]
    all_data_df.columns = ['date'] + [name + "_" + fix_string(c) if fix_string(c)[:8] != "unnamed:" else name for c in all_data_df.columns[1:]]
    
    # change the time column to a timestamp
    all_data_df.date = pd.to_datetime(all_data_df.date).dt.date
    
    # remove this file from the downloads folder
    os.remove(file_path)

    for i in range(1, num_csvs):
        
        # read in the file
        file_path = os.path.join(downloads_folder, 'bitcoinity_data ({}).csv'.format(i))
        df = pd.read_csv(file_path)

        # fix the column names
        name = metric_names[i]
        df.columns = ['date'] + [name + "_" + fix_string(c) if fix_string(c)[:8] != "unnamed:" else name for c in df.columns[1:]]
        
        # change the time column to a timestamp
        df.date = pd.to_datetime(df.date).dt.date
    
        # join this to the other data
        all_data_df = pd.merge(all_data_df, df, on="date", how="outer")
        
        # remove this file from the downloads folder
        os.remove(file_path)
        
    return all_data_df


def export_daily_data_from_all_days(driver, attempt=1):
    
    """
    1. Ensure that the chart shows 'All' the data - goes as far back as the chart goes
    2. Ensure the time frame for the chart is 'day' - one data point for each day
    """
    
    # do no re-attempt more than twice
    if attempt == 3:
        sys.exit()

    try:
        # click the button to show all data from bitcoins conception
        driver.find_element_by_xpath('//*[@id="chart"]/div[3]/div[1]/span/a[11]').click()
        
        # click the button to show the interval in a day time timeframe
        driver.find_element_by_xpath('//*[@id="chart"]/div[3]/div[2]/span/a[5]').click()
        
        # give the website time to load this data
        time.sleep(15)
            
        # click the button to export the data
        driver.find_element_by_xpath('//*[@id="chart"]/div[9]/div[2]/table/tbody/tr[5]/td[2]/a[1]').click()

        '''
        # click the button to show all data from bitcoins conception
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="chart"]/div[3]/div[1]/span/a[11]'))).click()
    
        # click the button to show the interval in a day time timeframe
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="chart"]/div[3]/div[2]/span/a[5]'))).click()

        time.sleep(10)

        # click the button to export the data
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="chart"]/div[9]/div[2]/table/tbody/tr[5]/td[2]/a[1]'))).click()
        '''

    except:
        print("       Re-trying data scrape")
        # if exporting the CSV took too long and timed out, re-attempt this function
        export_daily_data_from_all_days(driver, attempt=attempt+1)


def fix_string(string):
    
    """
    Take in a string and:
        1. Turn it to lowercase
        2. Replace spaces with underscores
        3. Remove other special characters
    """
    
    replace_dict = {' ': '_',
                    '/': '_and_',
                    '-': '_',
                   }
    
    s = string.lower()
    
    for k,v in replace_dict.items():
        s = s.replace(k, v)
        
    return s


def scrape_and_export_the_bitcoin_data(driver, block_num, range_vals, exclude_set):
    
    """
    Scrape the data in this sector (either Market data or Blockchain data)
    Each different piece of data is exported to a CSV in the downlaods folder
    These csv's are then read back into a table and deleted from the downloads folder using another function
    """
    
    list_of_names = []
    count_exports = 0
    for i in tqdm(range_vals):
        
        if i in exclude_set:
            continue 
            
        element_basic_xpath = '/html/body/div[2]/div[1]/div[1]/ul[{}]/li[{}]'.format(block_num, i)

        # get the name of the data metric
        metric_name = driver.find_element_by_xpath(element_basic_xpath).text
        
        print("    -", metric_name)
            
        # fix this name
        name = fix_string(metric_name)
        list_of_names.append(name)
        
        # call up a chart for this metric by clicking the button for this
        driver.find_element_by_xpath(element_basic_xpath + '/a').click()
        time.sleep(5)

        '''
        # call up a chart for this metric by clicking the button for this
        try:
            WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, element_basic_xpath+'/a'))).click()
        except TimeoutException:
            print("Timed out when calling up the {} chart".format(metric_name))
            sys.exit(1)
        '''

        # export this data
        export_daily_data_from_all_days(driver)
        count_exports += 1
        
    return count_exports, list_of_names


def get_market_data(driver, downloads_folder):
    
    """
    Scrape the data that falls under the 'Markets' heading on the bitcoinity.org website
    This data relates to bitcoin as a traded asset
    """
    
    print("Scraping the bitcoin market data")
    block_num = 1
    
    # there are 13 metrics in this block
    range_vals = range(2, 13)
    
    # exclude 'Price + Volume', 'Arbitrage', 'Combined Order Book'
    exclude_set = {5, 9, 10}
    
    # iterate through and scrape the different market data and export them to a CSV
    num_scraped, metric_names = scrape_and_export_the_bitcoin_data(driver, block_num, range_vals, exclude_set)
    
    # read these csv's back in and put them into a dataframe
    market_data_df = read_exported_csvs_into_df(num_scraped, metric_names, downloads_folder)
    
    return market_data_df


def get_blockchain_data(driver, downloads_folder):
    
    """
    Scrape the data that falls under the 'Blockchain' heading on the bitcoinity.org website
    This data relates to bitcoin as a network
    """
    
    print("Scraping the bitcoin blockchain data")
    block_num = 2
    
    # there are 8 metrics in this block
    range_vals = range(1, 8)
    
    # exclude 
    exclude_set = {4}
    
    # iterate through and scrape the different market data and export them to a CSV
    num_scraped, metric_names = scrape_and_export_the_bitcoin_data(driver, block_num, range_vals, exclude_set)
    
    # read these csv's back in and put them into a dataframe
    blockchain_data_df = read_exported_csvs_into_df(num_scraped, metric_names, downloads_folder)
    
    return blockchain_data_df


def scrape_bitcoinity_data(data_directory, config_variables, merge_exchanges):
    
    """
    Open a Google Chrome tab on the bitcoinity.org website and scrape the data
    Return a table containing the full scraped data
    """
    
    # unpack the variables needed for this scrape from the configuration dictionary
    chromedriver_location = config_variables["chromedriver_location"]
    downloads_folder = config_variables["downloads_folder"]
    bitcoinity_features_list = config_variables["bitcoinity_features_list"]

    # open the browser
    bitcoinity_url = 'https://data.bitcoinity.org'
    driver = webdriver.Chrome(chromedriver_location)
    driver.get(bitcoinity_url)
    
    # scrape data
    market_data = get_market_data(driver, downloads_folder)
    blockchain_data = get_blockchain_data(driver, downloads_folder)
    
    # merge these into one dataframe
    full_df = pd.merge(market_data, blockchain_data, on="date", how="outer")
    
    # make the date column a datetime and order the rows
    full_df["date"] = pd.to_datetime(full_df["date"])
    ordered_full_df = full_df.sort_values(["date"]).reset_index(drop=True)
    
    # change the other columns to numeric
    for col in ordered_full_df.drop(columns=["date"]):
        ordered_full_df[col] = pd.to_numeric(ordered_full_df[col])

    # merge the data values from the various exchanges into one column for each feature
    ordered_merged_cols_df = merge_different_exchanges_into_one_col(ordered_full_df, bitcoinity_features_list)
    
    # output the full and merged data to csv's
    ordered_full_df.to_csv(os.path.join(data_directory, "full_bitcoinity_internal_data.csv"), index=False)
    ordered_merged_cols_df.to_csv(os.path.join(data_directory, "merged_cols_bitcoinity_internal_data.csv"), index=False)

    final_df = ordered_merged_cols_df if merge_exchanges else ordered_full_df

    # print the date range of this data
    get_date_range_of_data(final_df)

    return final_df


# ================================================
# |                 BITINFOCHARTS                |
# ================================================


def extract_the_chart_values_from_the_html_string(html_string):
    
    try:
        val_start = html_string.index('[[')
    except:
        print("'[[' is not in the script")
        sys.exit(0)

    try:
        val_end = html_string.index(']]')
    except:
        print("']]' is not in the script")
        sys.exit(0)

    values = html_string[val_start+2:val_end]

    replace_dict = {'new Date("': '',
                    '"),': "|",
                    '],[': ',',
                    #'null': [None],
                   }
    
    for k, v in replace_dict.items():
        values = values.replace(k, v)
    
    return values


def generate_urls(part_url, features_list, indicator_list, period_list, scrape_technichal_indicators):
    
    """
    Generate a list of the URL's to each feature that we have to scrape.
    Each feature has its own unique URL so this function generates them so that we can iterate over them and scrape the data.
    
    Params:
        part_url
        features_list
        indicator_list
        period_list
        scrape_technichal_indicators
        
    Returns:
        tuple - (A list of the names of each feature, A list of the URL's to where we can scrape these features)
    """
    
    url_list = []
    feature_name_list = []
    
    # Generate the url structure for all the feature values
    for feature in features_list:
        
        # ==== Get the URLs for the raw values ====
        
        # get the URL
        url = '{}/{}-btc.html'.format(part_url, feature)
        url_list.append(url)

        feature_name_list.append(feature)

        # ==== Get the URLs for the technichal indicator values ====
        if scrape_technichal_indicators:
            for indicator in indicator_list:
                for period in period_list:

                    # Get the URL
                    url = '{}/{}-btc-{}{}.html'.format(part_url, feature, indicator, period)
                    url_list.append(url)

                    feature_name_list.append(feature + period + indicator)

    try:
        assert len(feature_name_list) == len(url_list)
    except:
        print("The number of features we have in oru list doesn't match the number of URL's we have")
        print("See the 'generate_urls' function")
        sys.exit(0)
    
    return feature_name_list, url_list


def scrape_bitinfocharts_data(data_folder, config_variables, start_date='2000/01/01', end_date='2050/01/01', include_technichal_indicators=True, output_features=False):
    
    '''
    Scrape the data from the bitinfocharts website, this can be all bitcoin data or just data between certain dates

    Parameters:
        config_variables:
        data_folder:
        start_date: date string - A string of the minimum date to be in the scraped data - format = YYYY/MM/DD
        end_date: date string - A string of the maximum date to be in the scraped data - format = YYYY/MM/DD
        include_technichal_indicators: Boolean - Whether we want the technichal indicators to be scraped or not
        output_features: Boolean - Whether we want a csv of each feature's data to be output after it is scraped
        
    Returns:
        Pandas Dataframe containg all the scraped data
    '''
    
    # define the location to output the individual scraped features if specified
    feature_output_directory = os.path.join(data_folder, "bitinfocharts_features")

    # unpack the variables needed for this scrape from the configuration dictionary
    features_list = config_variables["bitinfocharts_features_list"]
    indicator_list = config_variables["technical_indicator_list"] 
    period_list = config_variables["technichal_indicator_time_period_list"]

    # generate the url's needed to scrape the data from this website
    start_of_bitinfocharts_url = 'https://bitinfocharts.com/comparison'
    feature_name_list, url_list = generate_urls(start_of_bitinfocharts_url, features_list, indicator_list, period_list, include_technichal_indicators)
    
    num_features = len(feature_name_list)
    
    # The most important thing getting the data from the website is: DONT ABUSE IT
    # You might be IP banned for requesting a lot
    
    for i in tqdm(range(len(feature_name_list))):
        
        feature = feature_name_list[i]
        url = url_list[i]
        
        # access the website
        session = requests.Session()
        retry = Retry(connect=10, backoff_factor=3)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        page = session.get(url)
        
        # parse the html
        soup = BeautifulSoup(page.content, 'html.parser')
        
        # get the time series values - extract them from the messy string
        html_string = str(soup.find_all('script')[4])
        values = extract_the_chart_values_from_the_html_string(html_string)
            
        # crate a dataframe of this data - |date|value|
        list_of_all_vals = [str_val.split('|') for str_val in values.split(',')]
        feature_df = pd.DataFrame(list_of_all_vals, columns=['date', feature])

        # Get a subset of this data using the specified date range        
        subset_feature_df = feature_df.loc[(feature_df['date'] >= start_date) & (feature_df['date'] <= end_date)]
        
        # add this df to the df with the rest of the features
        if i == 0:
            # create a datafame to hosue all the features
            all_features_df = subset_feature_df
            
        else:
            all_features_df = pd.merge(all_features_df, subset_feature_df, on="date", how="outer")

        if output_features:
            # output this features data as a csv
            subset_feature_df.to_csv(os.path.join(feature_output_directory, feature + '.csv'), sep=',', columns=[feature], index=False)

    # make the date column a timestamp
    all_features_df["date"] = pd.to_datetime(all_features_df["date"])
    
    # change the other columns to numeric
    for col in all_features_df.drop(columns=["date"]):
        all_features_df[col] = pd.to_numeric(all_features_df[col].replace(['null'], [None]))

    # order the data by the date column
    sorted_df = all_features_df.sort_values('date').reset_index(drop=True)
    
    # output the data to a CSV
    sorted_df.to_csv(os.path.join(data_folder, "bitinfocharts_internal_data.csv"), index=False)

    # print the date range of this data
    get_date_range_of_data(sorted_df)

    return sorted_df



# ================================================
# |                 YAHOO FINANCE                |
# ================================================


def merge_stock_data_to_one_df(stock_name_to_data):

    full_df = pd.DataFrame()
    for stock_name, stock_df in stock_name_to_data.items():

        # add columns to full data table
        full_df = pd.merge(full_df, stock_df, left_index=True, right_index=True, how="outer")

    return full_df


def fill_in_weekend_values(df, stock_name):

    # find the max and min values in the dataframe
    min_date = min(df.index)
    max_date = datetime.now().date()
    date_range = pd.date_range(start=min_date, end=max_date)

    # iterate over this date range and add weekend values
    all_days_df = df.copy()
    for date in date_range:
        date = date.date()
        # if the market was closed on this day so there is no data  
        if date not in df.index:
            # create a data row with the values from the previous open day
            cols_in_df = [stock_name+"_open", stock_name+"_high", stock_name+"_low", stock_name+"_close"]
            row_dict = {}
            for col in cols_in_df:
                if col in all_days_df:
                    row_dict[col] = last_close
            if stock_name+"_volume" in all_days_df:
                row_dict[stock_name+"_volume"] = 0
            row = pd.DataFrame(row_dict, index=[date])

            # add this row to a dataframe wth all dates
            all_days_df = all_days_df.append(row)

        else:
            # this is a weekday
            last_close = df.loc[date, stock_name+"_close"]

    # return the dataframe sorted by the index
    return all_days_df.sort_index()


def scrape_stock_from_yahoo_finance(data_directory, name_to_ticker_map, start_date):

    # scrape one month prior to the start date - helps when filling the data
    y, m, d = start_date.split("-")
    new_start_date = "-".join([y if int(m) > 1 else str(int(y)-1), str(int(m)-1 if int(m) > 1 else 12), d])

    stock_name_to_data = {}
    for stock_name, ticker_val in tqdm(name_to_ticker_map.items()):
        
        # get ticker data
        ticker_data = yf.Ticker(ticker_val)
        
        # get historic data
        ticker_hist = ticker_data.history(start=new_start_date)

        # turn columns to lowercase, remove_spaces & add the stock name as a prefix
        ticker_hist.columns = [stock_name + "_" + c.lower().replace(' ', '_') for c in ticker_hist.columns]
        
        # remove the timestamp from the date index
        ticker_hist.index = ticker_hist.index.date
        ticker_hist.index = ticker_hist.index.rename("date")
        
        # remove columns that are all zeros or all 'nan'
        zero_cols = []
        for col in ticker_hist.columns:
            if ((ticker_hist[col] == 0).all()) or (is_nan(ticker_hist[col]).all()):
                zero_cols.append(col)
        non_zero_df = ticker_hist.drop(columns=zero_cols)
        
        # fill in the weekend values of this dataframe
        all_days_df = fill_in_weekend_values(non_zero_df, stock_name)

        # store in a dictionary
        stock_name_to_data[stock_name] = all_days_df

    # merge the data from these different tables into one dataframe
    merged_stock_df = merge_stock_data_to_one_df(stock_name_to_data)

    # remove the one month previous worth of data
    final_df = merged_stock_df.loc[datetime.strptime(start_date, "%Y-%m-%d").date():, :]

    # rename the index
    final_df.index.rename("date", inplace=True)

    # output the data to a CSV
    final_df.to_csv(os.path.join(data_directory, "yahoo_stock_data.csv"))

    return final_df


# ================================================
# |           FRED ECONOMIC INDICATORS           |
# ================================================


def get_data_from_quandl_page(driver):

    name_to_index = {}
    for i in range(1, 11):
        element_xpath = '/html/body/div[4]/div/div[1]/div/div/main/div/ul/li[{}]'.format(i)
        index_name = driver.find_element_by_xpath(element_xpath+'/dataset-card/dataset-header/a/h2').text
        fred_index_code = driver.find_element_by_xpath(element_xpath+'/dataset-card/dataset-summary/div/div[2]').text
        # /html/body/div[4]/div/div[1]/div/div/main/div/ul/li[1]/dataset-card/dataset-header/a/h2

        if 'Projection' in index_name:
            continue

        print(index_name, "-->", fred_index_code)
        
        name_to_index[index_name] = fred_index_code

    return name_to_index


def scrape_fed_economic_data_codes(config_variables):

    chromedriver_location = config_variables["chromedriver_location"]
    
    quandl_fred_url = 'https://www.quandl.com/data/FRED-Federal-Reserve-Economic-Data'
    driver = webdriver.Chrome(chromedriver_location)
    driver.get(quandl_fred_url)
    
    # iterate through each page and scrape the indicators details from them
    name_to_index = {}
    while True:
        try:
            # add the dictionary of values from this page to all the values
            name_to_index = {**name_to_index, **get_data_from_quandl_page(driver)}

            # click the 'next' button to go to the next page
            driver.find_element_by_xpath('/html/body/div[4]/div/div[1]/div/div/main/div/div[2]/button').click()

        except NoSuchElementException:
            break

    return name_to_index


def replace_invalid_vals_in_df(input_df):

    # define the values to replace
    replace_dict = {"NA": np.nan,
                    "None": np.nan,
                    "NaN": np.nan,
                    "none": np.nan,
                    "nan": np.nan,
                    "na": np.nan,
                   }

    # iterate through these alues and replace them
    df = input_df.copy()
    for k, v in replace_dict.items():
        df = df.replace(k, v)

    return df


def fill_in_days_between_economic_readings(input_df):

    # replace invalid values in the dataframe with NaNs
    df = replace_invalid_vals_in_df(input_df)

    # find the max and min values in the dataframe
    min_date = min(df.index)
    max_date = datetime.now().date()
    date_range = pd.date_range(start=min_date, end=max_date)

    # create a blank dataframe with a date for each day in this range
    date_for_each_day_df = pd.DataFrame(index=[d.date() for d in date_range])

    # add the input dataframe data to this daily dataframe
    full_df = pd.concat([date_for_each_day_df, df], axis=1)

    # fill the 'nan' values in this df by carrying the reading values forward till the next reading
    filled_df = full_df.fillna(method='ffill')

    return filled_df


def scrape_fred_indicators(data_directory, fred_indicator_tickers, start_date):
    
    # get todays date
    todays_date = datetime.today().strftime('%Y-%m-%d')

    # scrape three months prior to the start date - helps when filling the data
    y, m, d = start_date.split("-")
    new_month = int(m)-3 if int(m) > 3 else (12 if int(m) == 3 else (11 if int(m) == 2 else 10))
    new_start_date = "-".join([y if int(m) > 3 else str(int(y)-1), str(new_month), d])

    # iterate through the tickers and put them in a df
    fred_df = pd.DataFrame()
    for indicator_name, fred_ticker in tqdm(fred_indicator_tickers.items()):
        # read the data from the start date till todays date
        try:
            df = quandl.get("FRED/"+fred_ticker, start_date=new_start_date, end_date=todays_date)
        except NotFoundError:
            print("Couldn't find Quandl code '{}' - ({})".format(fred_ticker, indicator_name))

        # rename the column
        renamed_df = df.rename(columns={"Value": indicator_name})

        # fill in the values between the data readings
        filled_indicator_df = fill_in_days_between_economic_readings(renamed_df)

        # add this column to the final df
        fred_df = pd.concat([fred_df, filled_indicator_df], axis=1)
        
    # rename the index
    fred_df.index.rename("date", inplace=True)

    # remove the three months previous worth of data
    final_df = fred_df.loc[datetime.strptime(start_date, "%Y-%m-%d").date():, :]

    # output the data to a CSV
    final_df.to_csv(os.path.join(data_directory, "fred_economic_data.csv"))

    return final_df



# ================================================
# |     DB.NOMICS.WORLD ECONOMIC INDICATORS      |
# ================================================


def scrape_indicator(url):
    
    # get the data from the website
    web = requests.get(url)
    web_json = web.json()

    #print(web_json)

    # extract the needed info from this json dictionary
    periods = web_json['series']['docs'][0]['period']
    values = web_json['series']['docs'][0]['value']
    dataset = web_json['series']['docs'][0]['dataset_name']
    
    # create a dataframe containing the scraped data
    indicators = pd.DataFrame(values, index=periods)
    indicators.columns = [dataset]

    return indicators


def scrape_db_nomics_country_indicators(country_indicator_dict, countries_dict, start_date, fill_start_date):
    
    all_country_indicator_df = pd.DataFrame()
    failed_urls = []

    # scrape the indicators that need to be combined with the country codes
    with tqdm(total=(len(country_indicator_dict) * len(countries_dict))) as pbar:
        for indicator_name, ticker in country_indicator_dict.items():
            # iterate through each country
            for country_name, country_code in countries_dict.items():

                # define the indicator for this country
                ind_and_country_name = country_name + "_" + indicator_name
                ind_and_country_ticker = ticker + country_code

                # scrape the data
                url = "https://api.db.nomics.world/v22/series/{}?observations=1".format(ind_and_country_ticker)

                try:
                    df = scrape_indicator(url)
                except:
                    failed_urls.append(url)
                    pbar.update(1)
                    continue

                # turn the index to a datetime
                df.index = pd.to_datetime(df.index)
            
                # rename the column
                df.columns = [ind_and_country_name]

                # remove unneccesary dates - speed up computation
                subset_df = df.loc[datetime.strptime(fill_start_date, "%Y-%m-%d").date():, :]

                # check if the subsetted dataframe is now empty
                if subset_df.empty:
                    pbar.update(1)
                    continue

                # fill in the values between the data readings
                filled_df = fill_in_days_between_economic_readings(subset_df)

                # check if all values are now Nan in this dataframe
                if pd.DataFrame(filled_df.value_counts()).empty:
                    pbar.update(1)
                    continue

                # add this to a dataframe of all the data
                all_country_indicator_df = pd.concat([all_country_indicator_df, filled_df], axis=1)
                pbar.update(1)

    # rename the index
    all_country_indicator_df.index.rename("date", inplace=True)

    # remove the three months previous worth of data
    final_country_indicator_df = all_country_indicator_df.loc[datetime.strptime(start_date, "%Y-%m-%d").date():, :]

    return final_country_indicator_df, failed_urls


def scrape_db_nomics_standalone_indicators(indicator_dict, time_period_dict, start_date, fill_start_date):
    
    all_standalone_indicator_df = pd.DataFrame()

    # scrape the indicators that need to be combined with the country codes
    with tqdm(total=(len(indicator_dict) * len(time_period_dict))) as pbar:
        for indicator_name, ticker in indicator_dict.items():
            # iterate through each time period
            for period_name, period_val in time_period_dict.items():

                # define the indicator for this period
                ind_and_per_name = indicator_name + "_" + period_val
                ind_and_per_ticker = ticker.format(period_val)

                # scrape the data
                url = "https://api.db.nomics.world/v22/series/{}?observations=1".format(ind_and_per_ticker)
                df = scrape_indicator(url)

                # turn the index to a datetime
                df.index = pd.to_datetime(df.index)
            
                # rename the column
                df.columns = [ind_and_per_name]

                # remove unneccesary dates - speed up computation
                subset_df = df.loc[datetime.strptime(fill_start_date, "%Y-%m-%d").date():, :]

                # fill in the values between the data readings
                filled_df = fill_in_days_between_economic_readings(subset_df)

                # add this to a dataframe of all the data
                all_standalone_indicator_df = pd.concat([all_standalone_indicator_df, filled_df], axis=1)
                pbar.update(1)

    # rename tbe index
    all_standalone_indicator_df.index.rename("date", inplace=True)

    # remove the three months previous worth of data
    final_standalone_indicator_df = all_standalone_indicator_df.loc[datetime.strptime(start_date, "%Y-%m-%d").date():, :]

    return final_standalone_indicator_df


def scrape_db_nomics_economic_data(data_directory, start_date, standalone_indicator_dict, country_indicator_dict, countries_dict, time_periods_dict):

    # scrape three months prior to the start date - helps when filling the data
    y, m, d = start_date.split("-")
    new_month = int(m)-3 if int(m) > 3 else (12 if int(m) == 3 else (11 if int(m) == 2 else 10))
    fill_start_date = "-".join([y if int(m) > 3 else str(int(y)-1), str(new_month), d])

    # scrape the indicators that are standalone
    standalone_indicators = scrape_db_nomics_standalone_indicators(standalone_indicator_dict, time_periods_dict, start_date, fill_start_date)

    #print(standalone_indicators)

    # scrape the indicators that need to be combined with the country codes
    country_specific_indicators, failed_urls = scrape_db_nomics_country_indicators(country_indicator_dict, countries_dict, start_date, fill_start_date)

    #print(country_specific_indicators)

    # join these into one dataframe
    all_economic_df = pd.concat([standalone_indicators, country_specific_indicators], axis=1)

    # output the data to a CSV
    all_economic_df.to_csv(os.path.join(data_directory, "dbnomics_economic_data.csv"))

    return all_economic_df, failed_urls



# ================================================
# |              TWITTER SENTIMENT               |
# ================================================


def define_twitter_api():

    key = os.environ.get('TWITTER_API_KEY')
    secret_key = os.environ.get('TWITTER_API_SECRET_KEY')
    bearer_token = os.environ.get('TWITTER_API_BEARER_TOKEN')
    access_token = os.environ.get('TWITTER_API_ACCESS_TOKEN')
    secret_access_token = os.environ.get('TWITTER_API_SECRET_ACCESS_TOKEN')

    # create OAuthHandler object
    twitter_authentication = OAuthHandler(key, secret_key)

    # set access token and secret
    twitter_authentication.set_access_token(access_token, secret_access_token)

    # set up api using the authentication
    twitter_api = tweepy.API(twitter_authentication, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    return twitter_api


def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def get_tweet_sentiment(tweet):

    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''

    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))

    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def get_all_tweets_using_query(api, env_name, query, from_date, to_date, num_results=0):

    '''
    Main function to fetch tweets and parse them.
    '''

    # empty list to store parsed tweets
    tweets = []

    # call twitter api to fetch tweets
    if num_results:
        fetched_tweets = api.search_full_archive(environment_name="bitcoinTweets", query=query, fromDate=from_date, toDate=to_date, maxResults=num_results)
    else:
        fetched_tweets = api.search_full_archive(environment_name="bitcoinTweets", query=query, fromDate=from_date, toDate=to_date)

    for tweet in fetched_tweets:
        parsed_tweet = {'text': tweet.text, 'sentiment': get_tweet_sentiment(tweet.text)}

        # append parsed tweet list - ensure that retweets don't get through as duplicate
        if not (tweet.retweet_count > 0 and parsed_tweet in tweets):
            tweets.append(parsed_tweet)

    # return parsed tweets
    return tweets


def get_sentiment_breakdown_of_tweets(list_of_tweets_dicts, date):
    
    # picking positive and negative tweets from tweets
    pos_tweets = [tweet_dict for tweet_dict in list_of_tweets_dicts if tweet_dict['sentiment'] == 'positive']
    neg_tweets = [tweet_dict for tweet_dict in list_of_tweets_dicts if tweet_dict['sentiment'] == 'negative']
    neut_tweets = [tweet_dict for tweet_dict in list_of_tweets_dicts if tweet_dict['sentiment'] == 'neutral']

    # get the number of tweets with each sentiment
    num_total_tweets = len(list_of_tweets_dicts)
    num_pos_tweets = len(pos_tweets)
    num_neg_tweets = len(neg_tweets)
    num_neut_tweets = len(neut_tweets)

    # get the percentage of tweets each sentiment class has - if there was any tweets
    percent_pos = (num_pos_tweets/num_total_tweets) * 100 if num_total_tweets > 0 else 0
    percent_neg = (num_neg_tweets/num_total_tweets) * 100 if num_total_tweets > 0 else 0
    percent_neut = (num_neut_tweets/num_total_tweets) * 100 if num_total_tweets > 0 else 0

    # create a dictionary to house these derived values
    breakdown_dict = {"num_tweets"     : num_total_tweets,
                      "num_pos_tweets" : num_pos_tweets,
                      "num_neg_tweets" : num_neg_tweets,
                      "num_neut_tweets": num_neut_tweets,
                      "percent_pos"    : percent_pos,
                      "percent_neg"    : percent_neg,
                      "percent_neut"   : percent_neut,
                     }

    return pd.DataFrame(breakdown_dict, index=[date])


def turn_date_to_twitter_api_format(date):

    # extract the year month and day from the date
    str_date = str(date).split(" ")[0]
    year, month, day = str_date.split("-")

    # create a new string of this date
    new_date = year + month + day + ("0"*4)

    return new_date


def scrape_tweets_sentiment(query_words, start_date, config_variables):

    # define the twitter environment name
    env_name = config_variables["twitter_env_name"]

    # define the api to access twitter
    twitter_api = define_twitter_api()

    # get the range of dates we need to iterate through to scrape
    todays_date = datetime.today()
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    interval_size_in_days = int(str(todays_date - start_date_datetime).split(",")[0].split(" ")[0])

    # turn the query words into a query
    search_query = ' OR '.join([word if len(word.split(" ")) == 1 else '"'+word+'"' for word in query_words])

    # iterate through each day and calculate the sentiment of tweets for that day
    all_sentiment_dfs = pd.DataFrame()
    for days_since_start in tqdm(range(interval_size_in_days)):
        
        # define the day interval to look for bitcoin tweets in
        f_date = start_date_datetime + timedelta(days=days_since_start)
        t_date = f_date + timedelta(days=1)

        # turn the date to a string format for the twitter api
        fixed_date = turn_date_to_twitter_api_format(f_date)
        tommorows_date = turn_date_to_twitter_api_format(t_date)

        # calling function to get tweets
        tweets = get_all_tweets_using_query(twitter_api, env_name, query=search_query, from_date=fixed_date, to_date=tommorows_date)

        # get their sentiment breakdown
        sentiment_df = get_sentiment_breakdown_of_tweets(tweets, f_date)

        # add this sentiment df row to the other rows
        all_sentiment_dfs = pd.concat([all_sentiment_dfs, sentiment_df], axis=0)

    # set the index name to 'date'
    all_sentiment_dfs.index.rename("date", inplace=True)

    return all_sentiment_dfs



# ================================================
# |              TWITTER INFLUENCERS             |
# ================================================

def get_influencers_tweets(api, env_name, username, query, from_date, to_date, num_results=0):

    pass


def scrape_influencer_tweets(bitcoin_influencers, search_query, start_date, config_variables):

    # define the twitter environment name
    env_name = config_variables["twitter_env_name"]

    # define the api to access twitter
    twitter_api = define_twitter_api()

    # get the range of dates we need to iterate through to scrape
    todays_date = datetime.today()
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")

    # turn the date to a string format for the twitter api
    from_date = turn_date_to_twitter_api_format(start_date_datetime)
    to_date = turn_date_to_twitter_api_format(todays_date)

    # scrape the influencers tweets for that day
    for twitter_username in bitcoin_influencers:

        # calling function to get tweets
        tweets = get_influencers_tweets(twitter_api, env_name, username=twitter_username, query=search_query, from_date=from_date, to_date=to_date)


        break



# ================================================
# |            ELON MUSK TWITTER DATA            |
# ================================================

def get_all_musks_tweets(data_directory):

    # read in the csv where the tweets are saved
    tweets_full_df = pd.read_csv(os.path.join(data_directory, "musk_tweets_till_2021_data.csv"))

    # print the date range of the scrape
    min_date = min(tweets_full_df["date"])
    max_date = max(tweets_full_df["date"])
    print("Scraping tweets from '{}' --> '{}'".format(min_date.split(" ")[0], max_date.split(" ")[0]))

    # set up dataframe to store this tweets data
    tweet_df = pd.DataFrame(columns=["date", "tweet", "sentiment"])

    # iterate through each tweet and add it to our dataframe
    for index, row in tweets_full_df.iterrows():
        tweet_date = datetime. strptime(row["date"], '%Y-%m-%d %H:%M:%S').date()
        tweet = row["tweet"]
        row_df = pd.DataFrame({"date": tweet_date, "tweet": tweet, "sentiment": get_tweet_sentiment(tweet)}, index=[0])

        # add this tweet to the dataframe of all tweets
        tweet_df = tweet_df.append(row_df).reset_index(drop=True)

    print("  - {} tweets scraped".format(len(tweets_full_df)))

    return tweet_df


def subset_tweet_df_to_tweets_related_to_bitcoin(tweets_df, bitcoin_related_words):

    # iterate through the tweets to find the ones related to bitcoin
    indexes_to_keep = []
    for index, row in tweets_df.iterrows():
        for word in bitcoin_related_words:
            if word.lower() in row["tweet"].lower():
                indexes_to_keep.append(index)

    # return only the tweets with these words in them
    return tweets_df.loc[indexes_to_keep, :]


def batch_tweets_for_daily_sentiment_breakdown(tweets_df):

    all_days_sentiment_breakdown = pd.DataFrame()

    # iterate through each unique day that Elon tweeted
    for date in tweets_df["date"].unique():

        # get all tweets for this day
        one_days_tweets = tweets_df[tweets_df["date"] == date]

        tweets_dicts = []
        for i, r in one_days_tweets.iterrows():
            tweet_dict = {"sentiment": r["sentiment"]}
            tweets_dicts.append(tweet_dict)

        # get the sentiment breakdown of that days tweets
        day_sentiment_breakdown = get_sentiment_breakdown_of_tweets(tweets_dicts, date)

        # add this to all days breakdowns
        all_days_sentiment_breakdown = pd.concat([all_days_sentiment_breakdown, day_sentiment_breakdown], axis=0)

    return all_days_sentiment_breakdown


def fill_days_without_tweets(df, start_date):

    # find the max and min values in the dataframe
    max_date = datetime.now().date()
    date_range = pd.date_range(start=start_date, end=max_date)

    # create a blank dataframe with a date for each day in this range
    date_for_each_day_df = pd.DataFrame(index=[d.date() for d in date_range])

    # add the input dataframe data to this daily dataframe
    full_df = pd.concat([date_for_each_day_df, df], axis=1)

    # fill the 'nan' values in this df by carrying the reading values forward till the next reading
    filled_df = full_df.fillna(value=0)

    return filled_df


def get_musk_bitcoin_sentiment_data(data_directory, bitcoin_related_phrases, start_date):

    # get all musks tweets
    musk_tweets_df = get_all_musks_tweets(data_directory)

    # subset these tweets to only ones related to bitcoin
    musk_bitcoin_tweets_df = subset_tweet_df_to_tweets_related_to_bitcoin(musk_tweets_df, bitcoin_related_phrases)

    # get the sentiment breakdown of these tweets
    musk_daily_sentiment_breakdown = batch_tweets_for_daily_sentiment_breakdown(musk_bitcoin_tweets_df)

    # make sure the columns have musks name in it
    musk_daily_sentiment_breakdown.columns = ["musk_"+c for c in musk_daily_sentiment_breakdown]

    # fill in the days when he didn't have a bitcoin tweet
    filled_musk_daily_sentiment_df = fill_days_without_tweets(musk_daily_sentiment_breakdown, start_date)

    # rename the index to date
    filled_musk_daily_sentiment_df.index.rename("date", inplace=True)

    return filled_musk_daily_sentiment_df


# ================================================
# |             MERGING DATA SOURCES             |
# ================================================

def merge_dfs_on_col(list_of_dfs, col_to_join_on):

    # merge the dataframes on the specified column
    merged_df = list_of_dfs[0]
    for df in list_of_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=col_to_join_on, how="outer")
        
    # sort the df by the date
    sorted_merged_df = merged_df.sort_values(col_to_join_on).reset_index(drop=True)

    return sorted_merged_df


def merge_dfs_on_index(list_of_dfs):

    # merge the dataframes on the specified column
    merged_df = list_of_dfs[0]
    for df in list_of_dfs[1:]:
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how="outer")
        
    # return the dataframe sorted by the date
    return merged_df.sort_index()


def read_in_all_data_and_merge_it(data_directory, list_of_files, start_date):
    
    # create a df to store the data
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    merged_df = pd.DataFrame(index=pd.date_range(start=start_date, end=datetime.today()))
    merged_df.index = merged_df.index.rename("date")

    # iterate through each file
    for file in list_of_files:

        # read in the data
        filepath = os.path.join(data_directory, file)
        df = pd.read_csv(filepath)

        # set the date to its index
        df = df.set_index("date")

        # add this data to a dataframe of all the data
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how="outer")

    return merged_df

