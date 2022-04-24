
# general packages
import pandas as pd
from tqdm.auto import tqdm
import time
import os
import sys
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



# ================================================
# |                 CHECK NANS                   |
# ================================================

def get_cols_with_nan(df):

    nan_cols = []
    for col in df.columns:
        any_nan = df[col].isnull().values.any()
        if any_nan:
            nan_cols.append(col)

    return nan_cols


def find_col_nan_ranges(df):

    # get the columns that contain an 'nan' value
    nan_cols = sorted(get_cols_with_nan(df))
    print("{}/{} columns had a 'NaN' value in them".format(len(nan_cols), len(df.columns)))

    # ensure the index is the date
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    # create a df of whether the value is NaN or not
    is_null_df = pd.DataFrame(np.where(df[nan_cols].isnull(), 1, 0), columns=nan_cols, index=df.index)

    # Add a row at the beginning and and end of this df saying the day was not null
    is_null_df = is_null_df.append(pd.DataFrame(np.zeros((1, len(nan_cols))), columns=nan_cols, index=[min(is_null_df.index) - timedelta(days=1)])).sort_index()
    is_null_df = is_null_df.append(pd.DataFrame(np.zeros((1, len(nan_cols))), columns=nan_cols, index=[max(is_null_df.index) + timedelta(days=1)])).sort_index()

    # Define where the first and last Nan's appear
    null_start_df = pd.DataFrame(np.where(is_null_df > is_null_df.shift(1), 1, 0)[1:-1], columns=nan_cols, index=df.index)
    null_end_df = pd.DataFrame(np.where(is_null_df > is_null_df.shift(-1), 1, 0)[1:-1], columns=nan_cols, index=df.index)

    # Extract where the Nan's start and end for each column
    col_to_nan_dates = {}
    for col in tqdm(nan_cols):
        start_dates = null_start_df[null_start_df[col] == 1].index
        end_dates = null_end_df[null_end_df[col] == 1].index

        assert len(start_dates) == len(end_dates)

        col_to_nan_dates[col] = []
        for nan_start_date, nan_end_date in zip(start_dates, end_dates):
            col_to_nan_dates[col].append((str(nan_start_date.date()), str(nan_end_date.date())))

    return col_to_nan_dates


def print_nan_cols(nan_col_dates_dict, num_cols):

    # get a list of the columns
    nan_cols = list(nan_col_dates_dict.keys())

    # print these columns
    if num_cols == 1:
        for a in nan_cols:
            print(a)
    elif num_cols == 2:
        for a,b in zip(nan_cols[::2], nan_cols[1::2]):
            print('{:<50}{:<}'.format(a, b))
    elif num_cols == 3:
        for a,b,c in zip(nan_cols[::3], nan_cols[1::3], nan_cols[2::3]):
            print('{:40}{:<40}{:<}'.format(a, b, c))
    elif num_cols == 4:
        for a,b,c,d in zip(nan_cols[::4], nan_cols[1::4], nan_cols[2::4], nan_cols[3::4]):
            print('{:30}{:<30}{:<30}{:<}'.format(a, b, c, d))



# ================================================
# |            FINALISE SCRAPED DATA             |
# ================================================

def output_data_date_range_summary(df):

    """
    Take in a dataframe with a date as it's index and output the span of the non-null data in this dataframe

    Parameters:
        df (dataframe) : A pandas dataframe that has a "date" index

    Returns:
        None
    """

    # get the min and max dates in this df
    non_null_rows = df[df.notna().any(axis=1)]

    # Find the min and max date in this non-null df
    min_date = min(non_null_rows.index).date()
    max_date = max(non_null_rows.index).date()
    interval_size_in_days = int(str(max_date - min_date).split(",")[0].split(" ")[0])

    print("The data scraped is from '{}' to '{}' - During this {} day period, there were {} non-null rows".format(min_date, max_date, interval_size_in_days, len(non_null_rows)))


def re_index_df_to_have_row_for_each_date(df, start_date):

    # get yesterdays date as a string
    yesterday = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')

    # drop any data from the input df that appears before the start date
    if start_date < str(min(df.index)):
        df = df.loc[datetime.strptime(start_date, "%Y-%m-%d").date():, :]

    # drop any data from the input df for today
    if yesterday > str(min(df.index)):
        df = df.loc[:datetime.strptime(yesterday, "%Y-%m-%d").date(), :]

    # create an index going from the start date till yesterday
    index = pd.date_range(start_date, yesterday)

    # reindex the input df to have a row for each day between start date and yesterday
    return df.reindex(index, fill_value=np.nan)


def replace_invalid_vals_in_df(input_df):

    # define the values to replace
    replace_dict = {"NA": np.nan,
                    "None": np.nan,
                    "NaN": np.nan,
                    "none": np.nan,
                    "nan": np.nan,
                    "na": np.nan,
                    "null": np.nan,
                    "Null": np.nan,
                   }

    # iterate through these alues and replace them
    df = input_df.copy()
    for k, v in replace_dict.items():
        df = df.replace(k, v)

    return df


def finalise_df_types(input_df, start_date, date_colname="date"):

    # replace invalid values in the dataframe with NaNs
    df = replace_invalid_vals_in_df(input_df)

    # turn the index into a date column if the date column isn't in the dataframe
    if date_colname not in df.columns:
        df.index.names = [date_colname]
        df = df.reset_index()

    # change the data type of the columns (except date) to numeric
    for col in df.drop(columns=[date_colname]):
        df[col] = pd.to_numeric(df[col])
    
    # make the date column a datetime, order the rows, and set it as the index
    df[date_colname] = pd.to_datetime(df[date_colname])
    sorted_df = df.sort_values([date_colname]).reset_index(drop=True).set_index(date_colname)

    # Ensure there is a row for each day in this data, fill missing values with 'np.nan'
    all_dates_df = re_index_df_to_have_row_for_each_date(sorted_df, start_date)

    # drop any columns which are all 'NaN'
    subset_df = all_dates_df.drop(columns=all_dates_df.columns[all_dates_df.isna().all(axis=0)].to_list())

    # print the date range of this data
    output_data_date_range_summary(subset_df)

    return subset_df



# ================================================
# |               BITCOINITY.ORG                 |
# ================================================

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


def scrape_bitcoinity_data(data_directory, config_variables, start_date):
    
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
    
    # close the browser
    driver.close()

    # merge these into one dataframe
    full_df = pd.merge(market_data, blockchain_data, on="date", how="outer")
    
    # finalise the types of the datframe
    final_df = finalise_df_types(full_df, start_date, date_colname="date")

    # output the full data to a csv
    final_df.to_csv(os.path.join(data_directory, "bitcoinity_raw_internal_data.csv"), index_label="date")

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


def scrape_bitinfocharts_data(data_folder, config_variables, start_date, include_tech_ind=True, output_features=False):
    
    '''
    Scrape the data from the bitinfocharts website, this can be all bitcoin data or just data between certain dates

    Parameters:
        config_variables:
        data_folder:
        start_d: date string - A string of the minimum date to be in the scraped data - format = YYYY/MM/DD
        end_d: date string - A string of the maximum date to be in the scraped data - format = YYYY/MM/DD
        include_tech_ind: Boolean - Whether we want the technichal indicators to be scraped or not
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
    feature_name_list, url_list = generate_urls(start_of_bitinfocharts_url, features_list, indicator_list, period_list, include_tech_ind)
    
    # The most important thing getting the data from the website is: DONT ABUSE IT
    # You might be IP banned for requesting a lot
    
    for i, (feature, url) in tqdm(enumerate(zip(feature_name_list, url_list)), total=len(feature_name_list)):

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

        # add this df to the df with the rest of the features
        all_features_df = feature_df if (i == 0) else pd.merge(all_features_df, feature_df, on="date", how="outer")

        if output_features:
            # output this features data as a csv
            feature_df.to_csv(os.path.join(feature_output_directory, feature + '.csv'), sep=',', columns=[feature], index=False)

    # finalise the types of the datframe
    final_df = finalise_df_types(all_features_df, start_date, date_colname="date")

    # output the data to a CSV
    final_df.to_csv(os.path.join(data_folder, "bitinfocharts_raw_internal_data.csv"), index_label="date")

    return final_df



# ================================================
# |                 YAHOO FINANCE                |
# ================================================


def merge_individual_yahoo_data_to_one_df(stock_name_to_data):

    full_df = pd.DataFrame()
    for stock_name, stock_df in stock_name_to_data.items():

        # add columns to full data table
        full_df = pd.merge(full_df, stock_df, left_index=True, right_index=True, how="outer")

    return full_df


def scrape_yahoo_finance_data(data_directory, name_to_ticker_map, start_date, data_name):

    stock_name_to_data = {}
    for stock_name, ticker_val in tqdm(name_to_ticker_map.items()):
        
        # get ticker data
        ticker_data = yf.Ticker(ticker_val)
        
        # get historic data
        ticker_hist = ticker_data.history(start=start_date)

        # turn columns to lowercase, remove_spaces & add the stock name as a prefix
        ticker_hist.columns = [stock_name + "_" + c.lower().replace(' ', '_') for c in ticker_hist.columns]
        
        # remove columns that are all zeros or all 'nan'
        zero_cols = []
        for col in ticker_hist.columns:
            if ((ticker_hist[col] == 0).all()) or (ticker_hist[col].isnull().all()):
                zero_cols.append(col)

        # store in a dictionary  
        stock_name_to_data[stock_name] = ticker_hist.drop(columns=zero_cols)

    # merge the data from these different tables into one dataframe using the date indexes
    merged_stock_df = merge_individual_yahoo_data_to_one_df(stock_name_to_data)

    # finalise the types of the datframe
    final_df = finalise_df_types(merged_stock_df, start_date, date_colname="date")

    # output the data to a CSV
    final_df.to_csv(os.path.join(data_directory, "yahoo_{}_data.csv".format(data_name)), index_label="date")

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


def scrape_fred_indicators(data_directory, fred_indicator_tickers, start_date):
    
    # define the quandl API key
    quandl.ApiConfig.api_key = os.environ.get('QUANDL_API_KEY')

    # get todays date
    todays_date = datetime.today().strftime('%Y-%m-%d')

    # iterate through the tickers and put them in a df
    fred_df = pd.DataFrame()
    for indicator_name, fred_ticker in tqdm(fred_indicator_tickers.items()):
        # read the data from the start date till todays date
        try:
            df = quandl.get("FRED/"+fred_ticker, start_date=start_date, end_date=todays_date)
        except NotFoundError:
            print("Couldn't find Quandl code '{}' - ({})".format(fred_ticker, indicator_name))

        # rename the column
        indicator_df = df.rename(columns={"Value": indicator_name})

        # add this column to the final df
        fred_df = pd.concat([fred_df, indicator_df], axis=1)

    # finalise the types of the datframe
    final_df = finalise_df_types(fred_df, start_date, date_colname="date")

    # output the data to a CSV
    final_df.to_csv(os.path.join(data_directory, "fred_economic_data.csv"), index_label="date")

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


def scrape_db_nomics_country_indicators(country_indicator_dict, countries_dict):
    
    # scrape the indicators that need to be combined with the country codes
    failed_urls = []
    all_country_indicator_df = pd.DataFrame()
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

                # turn the index to a datetime & rename the column
                df.index = pd.to_datetime(df.index)
                df.columns = [ind_and_country_name]

                # check if the subsetted dataframe is now empty or if all values in this dataframe are NaN
                if df.empty or pd.DataFrame(df.value_counts()).empty:
                    pbar.update(1)
                    continue

                # add this to a dataframe of all the data
                all_country_indicator_df = pd.concat([all_country_indicator_df, df], axis=1)
                pbar.update(1)

    return all_country_indicator_df, failed_urls


def scrape_db_nomics_standalone_indicators(indicator_dict, time_period_dict):

    # scrape the indicators that need to be combined with the country codes
    all_standalone_indicator_df = pd.DataFrame()
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

                # turn the index to a datetime & rename the column
                df.index = pd.to_datetime(df.index)
                df.columns = [ind_and_per_name]

                # add this to a dataframe of all the data
                all_standalone_indicator_df = pd.concat([all_standalone_indicator_df, df], axis=1)
                pbar.update(1)

    return all_standalone_indicator_df


def scrape_db_nomics_economic_data(data_directory, start_date, config_variables):

    # Extract the needed variables from the configuration dictionary
    time_periods_dict = config_variables["db_nomics_time_periods"]
    standalone_tickers = config_variables["db_nomics_eurostat_tickers_standalone"]
    country_codes_dict = config_variables["db_nomics_countries"]
    country_specific_tickers = config_variables["db_nomics_eurostat_tickers_for_countries"]

    # scrape the indicators that are standalone
    standalone_indicators = scrape_db_nomics_standalone_indicators(standalone_tickers, time_periods_dict)

    # scrape the indicators that need to be combined with the country codes
    country_specific_indicators, failed_urls = scrape_db_nomics_country_indicators(country_specific_tickers, country_codes_dict)

    # join these into one dataframe
    all_economic_df = pd.concat([standalone_indicators, country_specific_indicators], axis=1)

    # finalise the types of the datframe
    final_df = finalise_df_types(all_economic_df, start_date, date_colname="date")

    # output the data to a CSV
    final_df.to_csv(os.path.join(data_directory, "dbnomics_economic_data.csv"), index_label="date")

    return final_df, failed_urls



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


def scrape_tweets_sentiment(data_directory, start_date, config_variables):

    # define the twitter environment name
    env_name = config_variables["twitter_env_name"]

    # define the query words to use
    bitcoin_query_words = config_variables["twitter_bitcoin_query_words"]

    # define the api to access twitter
    twitter_api = define_twitter_api()

    # get the range of dates we need to iterate through to scrape
    todays_date = datetime.today()
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    interval_size_in_days = int(str(todays_date - start_date_datetime).split(",")[0].split(" ")[0])

    # turn the query words into a query
    search_query = ' OR '.join([word if len(word.split(" ")) == 1 else '"{}"'.format(word) for word in query_words])

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

    # save this to a csv
    all_sentiment_dfs.to_csv(os.path.join(data_directory, "twitter_bitcoin_sentiment_social_media_data.csv"), index=True)

    return all_sentiment_dfs



# ================================================
# |              TWITTER INFLUENCERS             |
# ================================================

def get_influencers_tweets(api, env_name, username, query, from_date, to_date, num_results=0):

    pass


def scrape_influencer_tweets(start_date, config_variables):

    # define the twitter environment name, bitcoin influencers account names, the query words to use
    env_name = config_variables["twitter_env_name"]
    bitcoin_influencers = config_variables["twitter_bitcoin_influencers"]
    bitcoin_query_words = config_variables["twitter_bitcoin_query_words"]

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
    print("Extracting Musks tweets from '{}' --> '{}'".format(min_date.split(" ")[0], max_date.split(" ")[0]))
    print("  - {} tweets extracted".format(len(tweets_full_df)))

    # extract the columns we want
    tweets_df = tweets_full_df[["date", "tweet"]]

    # calculate the tweet sentiment
    tweets_df["sentiment"] = tweets_df["tweet"].apply(get_tweet_sentiment)

    # remove the time from the date column
    tweets_df["date"] = pd.to_datetime(tweets_df["date"]).dt.date

    return tweets_df


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
        for _, row in one_days_tweets.iterrows():
            tweets_dicts.append({"sentiment": row["sentiment"]})

        # get the sentiment breakdown of that days tweets
        day_sentiment_breakdown = get_sentiment_breakdown_of_tweets(tweets_dicts, date)

        # add this to all days breakdowns
        all_days_sentiment_breakdown = pd.concat([all_days_sentiment_breakdown, day_sentiment_breakdown], axis=0)

    return all_days_sentiment_breakdown


def get_musk_bitcoin_sentiment_data(data_directory, start_date, config_variables):

    # Define the bitcoin related query words to look at
    bitcoin_related_phrases = config_variables["twitter_bitcoin_query_words"]

    # get all musks tweets
    musk_tweets_df = get_all_musks_tweets(data_directory)

    # subset these tweets to only ones related to bitcoin
    musk_bitcoin_tweets_df = subset_tweet_df_to_tweets_related_to_bitcoin(musk_tweets_df, bitcoin_related_phrases)

    # get the sentiment breakdown of these tweets
    musk_daily_sentiment_breakdown = batch_tweets_for_daily_sentiment_breakdown(musk_bitcoin_tweets_df)

    # make sure the columns have musks name in it
    musk_daily_sentiment_breakdown.columns = ["musk_"+c for c in musk_daily_sentiment_breakdown]

    # finalise the types of the datframe
    final_df = finalise_df_types(musk_daily_sentiment_breakdown, start_date, date_colname="date")

   # save this to a csv
    final_df.to_csv(os.path.join(data_directory, "musk_bitcoin_sentiment_social_media_data.csv"), index_label="date")

    return final_df

