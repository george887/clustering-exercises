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
