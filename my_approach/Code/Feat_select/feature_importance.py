
# General packages
import pandas as pd

# Packages for feature importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



def get_cols_with_nan(df):

    nan_cols = []
    for col in df.columns:
        any_nan = df[col].isnull().values.any()
        if any_nan:
            nan_cols.append(col)

    return nan_cols


def drop_nan_cols(df):

    nan_cols = get_cols_with_nan(df)
    print("\nDropping the following 'Nan' columns:\n", nan_cols, "\n")

    dropped = df.copy()
    return dropped.drop(columns=nan_cols)


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


def use_rand_forest_to_get_feature_importance(reg_or_clas, feature_df, y, random_seed):

    # fit the RFC model
    if reg_or_clas == 'reg':
        rf_final = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
    if reg_or_clas == 'clas':
        rf_final = RandomForestClassifier(random_state=random_seed, n_jobs=-1)
    rf_final.fit(feature_df, y)

    # make a dataframe of the feature importance
    feature_importance_dict = {"feature": feature_df.columns, "importance": rf_final.feature_importances_}
    feature_importance_df = pd.DataFrame(feature_importance_dict).sort_values(["importance"], ascending=False).reset_index(drop=True)

    return feature_importance_df

