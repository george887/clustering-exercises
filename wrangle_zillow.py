import pandas as pd
import numpy as np
# ENV has credentials needed to access the SQL database. Gitignore should be created before to prevent your info to be compromised
from env import host, user, password
# OS Looks to see if data is stored locally
import os
# allows us to ignore any warnings
import warnings
warnings.filterwarnings("ignore")

########################## Establishing connection ###########################
# establish mysql connection
def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

########################## Creating function to get data ######################
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    # Selecting all data in the properties_2017 table
    sql_query = '''  select * from properties_2017
                join predictions_2017 using (parcelid)
                left join airconditioningtype using (airconditioningtypeid)
                left join architecturalstyletype using (architecturalstyletypeid)
                left join buildingclasstype using (buildingclasstypeid)
                left join heatingorsystemtype using (heatingorsystemtypeid)
                left join propertylandusetype using (propertylandusetypeid)
                left join storytype using (storytypeid)
                left join typeconstructiontype using (typeconstructiontypeid)
                left join unique_properties using (parcelid)
                where latitude is not null and longitude is not null;
                '''

    # The pandas read_sql function allows us to create a df with the afformentioned sql querry    
    df = pd.read_sql(sql_query, get_connection('zillow'))

    # Converts the df into a csv
    df.to_csv('zillow_df.csv')

    # This prevents any duplicated columns. The ~ allows to return the unique columns. A boolean array is created
    # and only falses are returned
    df = df.loc[:,~df.columns.duplicated()]

    return df

def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database if cached == False, a csv is created
    returning the df. If cached == True, the function reads in the zillow df from a csv file & returns df
    '''
    # This runs if there is no csv containing the zillow data
    if cached or os.path.isfile('zillow_df.csv') == False:

        # Converts the df into a csv
        df = new_zillow_data()

    else:

        # If the csv was stored locally, the csv will return the df
        df = pd.read_csv('zillow_df.csv', index_col=0)

    return df

#################### Prepare ##################

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


def missing_values(df):
    ''' This function will calculate values of missing rows and return a missing rows df
    '''
    # Gives value counts of missing rows
    missing_row_value = df.isnull().sum()
    # Gives the percentage of rows missing
    percent_row_missing = round(df.isnull().sum()/len(df),2)*100
    # Creates a new df for the missing rows and percent missing
    missing_df = pd.DataFrame({'missing_rows' : missing_row_value, 'percent_missing' : percent_row_missing})
    return missing_df

def missing_cols(df):
    ''' This function will calculate values of missing columns and return a missing columns df
    '''
    # df.loc[ : ].count() means we're looking at every row to count the number of null values in each row 
    # .isna() shows if there are booleans if there is nulls. .any() looks for true values from the isna()
    # summing the count of trues with .count()
    missing_cols = df.loc[:, df.isna().any()].count()
    # len(df.index) shows the number of rows
    percent_cols_missing = round(df.loc[:, df.isna().any()].count()/ len(df.index) *100 ,2)
    missing_cols_df = pd.DataFrame({'missing_columns' : missing_cols, 'percent_columns_missing' : percent_cols_missing})
    return missing_cols_df

def single_unit_properties(df):
    # Creating df to meet criteria of single unit homes
    unitcnt_df = df[df.unitcnt == 1]
    bedroom_df = df[df.bedroomcnt > 0]
    bathroom_df = df[df.bathroomcnt > 0]
    # using | == (or) to filter properties based on land use type id
    single_prop_df = df[(df.propertylandusetypeid == 261) | (df.propertylandusetypeid == 263) | (df.propertylandusetypeid == 264) \
                 | (df.propertylandusetypeid == 266) | (df.propertylandusetypeid == 270) | (df.propertylandusetypeid == 273) \
                 | (df.propertylandusetypeid == 274) | (df.propertylandusetypeid == 275) | (df.propertylandusetypeid == 279)]
    # Concat all the df together. Dropping duplicate 'ids' as each property has a unique id.
    # reset index in case of multiple indexes being created
    single_unit_df = pd.concat([unitcnt_df, bedroom_df, bathroom_df, single_prop_df]).drop_duplicates('id').reset_index(drop=True)
    return single_unit_df

def handle_missing_values(df, prop_required_column, prop_required_row):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh = threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


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

def add_scaled_columns(train, validate, test, scaler = MinMaxScaler(), columns_to_scale):
    """This function scales the mall customers data"""
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
