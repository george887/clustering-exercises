import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from wrangle_zillow import get_mall_data, get_zillow_data
import warnings
warnings.filterwarnings("ignore")

#################### Zillow prepare #####################

def data_prep(df, cols_to_remove=[], prop_required_column = .6, prop_required_row = .75):
    '''
    This function removes columns and rows below the threshold
    '''
    df = df.drop(columns=cols_to_remove)  
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

    def remove_columns(df, cols_to_remove): 
    '''This function removes columns
    ''' 
    df = df.drop(columns=cols_to_remove)
    return df
    
    # Remove rows & columns based on a minimum percentage of values available for each row/columns:
    def handle_missing_values(df, prop_required_column = .6, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

    df = df.drop(columns=cols_to_remove)  
    df = handle_missing_values(df, prop_required_column, prop_required_row)

    return df

# Call the function like below
#f = prepare.data_prep(
#     df,
#     cols_to_remove=[],
#     prop_required_column=.6,
#     prop_required_row=.75

############## Mall Customers prepare ##############
def prep_mall_data(df):
    '''
    Takes the acquired mall data from wrangle_zillow, does data prep, and returns
    train, test, and validate data splits.
    '''
    # use pd.get_dummies to make gender a numeric
    dummies_df = pd.get_dummies(df[['gender']], drop_first=True)

    # renaming the new column to read easier in df
    dummies_df = dummies_df.rename(columns={'gender_Male': 'gender_male'})

    # concat the orginal df with the dummies
    df = pd.concat([df,dummies_df], axis=1)

    # dropping columns that serve no purpose to explore the data
    df = df.drop(columns=['customer_id', 'gender'])

    # splitting the data to explore and run analysis
    train_and_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.3, random_state=123)
    
    return train, test, validate

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))


def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df

#Now we can see what the outliers in our data look like:
#outlier_cols = [col for col in df if col.endswith('_outliers')]
#for col in outlier_cols:
    #print('~~~\n' + col)
    #data = df[col][df[col] > 0]
    #print(data.describe())
