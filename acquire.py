import pandas as pd
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

########################## Creating function to get zillow data ######################
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

    #################### Acquire Mall Customers Data ##################
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
