import pandas as pd
import numpy as np
# ENV has credentials needed to access the SQL database. Gitignore should be created before to prevent your info to be compromised
from env import host, user, password
# OS Looks to see if data is stored locally
import os
# allows us to ignore any warnings
import warnings
warnings.filterwarnings("ignore")
# imports the train test split
from sklearn.model_selection import train_test_split

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

def single_unit_properties(df):
    '''This function will filter single unit properties, fillna's, drop unwanted columns, and replace features
    '''
    df = df[df.propertylandusetypeid.isin([260,261,262,279])]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    df = df[(df.bedroomcnt < 6) & (df.bathroomcnt < 5)]
    df.unitcnt = df.unitcnt.fillna(1)
    df = df[df.unitcnt == 1.0]
    df = df.drop(columns=["propertylandusetypeid", "heatingorsystemtypeid", 'propertyzoningdesc', 'calculatedbathnbr', "id", "id.1"])
    df['heatingorsystemdesc'].replace(np.nan, 'none', inplace=True)
    return df

# How to call function df= single_unit_properties(df)

def handle_missing_values(df, prop_required_column = .60, prop_required_row = .60):
    ''' This function will drop na's when the required thresh is not met'''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# How to call function df=handle_missing_values(df, prop_required_column = .60, prop_required_row = .60)

def impute_missing_values(df):
    '''This function will split the data into train, validate and test data frames. I imputed the missing values of 
    the list of features (Categorical/Discrete) using the mode or most common
    '''

    train_and_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.3, random_state=123)
    
    cols1 = [
    "buildingqualitytypeid",
    "regionidcity",
    "regionidzip",
    "yearbuilt",
    "regionidcity",
    "censustractandblock"
    ]

    for col in cols1:
        mode = int(train[col].mode()) # I had some friction when this returned a float (and there were no decimals anyways)
        train[col].fillna(value=mode, inplace=True)
        validate[col].fillna(value=mode, inplace=True)
        test[col].fillna(value=mode, inplace=True)

    return train, validate, test

# How to call function train, validate, test =impute_missing_values(df)

def impute_missing_values_1(train, validate, test):
    '''This function will split the data into train, validate and test data frames. I imputed the missing values of 
    the list of features (Continuous columns) using the median
    '''

    cols = [
        "structuretaxvaluedollarcnt",
        "taxamount",
        "taxvaluedollarcnt",
        "landtaxvaluedollarcnt",
        "structuretaxvaluedollarcnt",
        "finishedsquarefeet12",
        "calculatedfinishedsquarefeet",
        "fullbathcnt",
        "lotsizesquarefeet"
    ]

    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        validate[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)
        
    return train, validate, test

# How to call function train, validate, test = impute_missing_values_1(train, validate, test)