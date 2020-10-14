import pandas as pd
import numpy as np
# ENV has credentials needed to access the SQL database. Gitignore should be created before to prevent your info to be compromised
from env import host, user, password
# OS Looks to see if data is stored locally
import os
# allows us to ignore any warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#################### Mall Customers Data #####################

def new_mall_data():
    ''' This function will get all the mall data in the SQL database and return a df'''
    # Query to get data from the customers table
    sql_query = 'SELECT * FROM customers'
     # The pandas read_sql function allows us to create a df with the afformentioned sql querry    
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    # Converts the df into a csv
    df.to_csv('mall_customers.csv')
    return df

def get_mall_data(cached=False):
    '''
    This function reads in mall customers data from Codeup database if cached == False, a csv is created
    returning the df. If cached == True, the function reads in the mall customers df from a csv file & returns df
    '''
    # This runs if there is no csv containing the zillow data
    if cached or os.path.isfile('mall_customers.csv') == False:

        # Converts the df into a csv
        df = new_mall_data()

    else:

        # If the csv was stored locally, the csv will return the df
        df = pd.read_csv('mall_customers.csv', index_col=0)

    return df

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


def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    '''This function scales the mall customers data'''
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test


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
