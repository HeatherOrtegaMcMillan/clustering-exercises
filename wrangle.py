#~#~#~#~#~#~# Zillow Clustering Wrangling Functions #~#~#~#~#~#~#

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~

################ Imports ################
import pandas as pd
import numpy as np
from env import host, password, user
import os
import scipy.stats as stats

from sklearn.model_selection import train_test_split

###################### Getting database Url ################
def get_db_url(db_name, user=user, host=host, password=password):
    """
        This helper function takes as default the user host and password from the env file.
        You must input the database name. It returns the appropriate URL to use in connecting to a database.
    """
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

######################### get generic data #########################
def get_any_data(database, sql_query):
    '''
    put in the query and the database and get the data you need in a dataframe
    '''

    return pd.read_sql(sql_query, get_db_url(database))

######################### get Zillow Data #########################
def get_zillow_data():
    '''
    This function reads in Zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    sql_query = """
                SELECT parcelid, airconditioningtypeid, airconditioningdesc, architecturalstyletypeid, architecturalstyledesc,
                bathroomcnt, bedroomcnt, buildingclasstypeid, buildingclassdesc, buildingqualitytypeid,
                decktypeid, calculatedfinishedsquarefeet, fips, fireplacecnt, fireplaceflag, garagecarcnt, garagetotalsqft,
                hashottuborspa, latitude, longitude, lotsizesquarefeet, poolcnt, poolsizesum, propertycountylandusecode,
                propertylandusetypeid, propertylandusedesc, propertyzoningdesc, rawcensustractandblock, 
                regionidcity, regionidcounty, regionidneighborhood, roomcnt, threequarterbathnbr, typeconstructiontypeid, typeconstructiondesc, unitcnt, yearbuilt, numberofstories, structuretaxvaluedollarcnt, taxvaluedollarcnt, assessmentyear, 
                landtaxvaluedollarcnt, taxamount, censustractandblock, logerror, transactiondate 
                FROM properties_2017 AS p
                JOIN predictions_2017 USING (parcelid)
                INNER JOIN (SELECT parcelid, MAX(transactiondate) AS transactiondate
                FROM predictions_2017
                GROUP BY parcelid) 
                AS t USING (parcelid, transactiondate)
                LEFT JOIN airconditioningtype USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
                LEFT JOIN storytype USING (storytypeid)
                LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL 
                AND transactiondate LIKE '2017%%';
                """
    if os.path.isfile('zillow_data.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = pd.read_sql(sql_query, get_db_url('zillow'))
        
        # Cache data
        df.to_csv('zillow_data.csv')

    return df

#################################### Look at nulls ####################################

# I saw this on a kaggle post. This is the credit that author gave.
# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 
# One of the best notebooks on getting started with a ML problem.

def missing_values_table(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values, 
    and the percent of that column that has missing values
    '''
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
        
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


####################################### Overview Function #######################################
def overview(df, thresh = 10):
    '''
    This function takes in a dataframe and prints out useful things about each column.
    Unique values, value counts for columns less than 10 (can be adjusted with optional argument thresh)
    Whether or not the row has nulls
    '''
    # create list of columns
    col_list = df.columns
    
    # loop through column list
    for col in col_list:
        # seperator using column name
        print(f'============== {col} ==============')
        
        # print out unique values for each column
        print(f'# Unique Vals: {df[col].nunique()}')
        
        # if number of things is under or equal to the threshold  print a value counts
        if df[col].nunique() <= thresh:
            print(df[col].value_counts(dropna = False))
            
        # if the number is less than 150 and not an object, bin it and do value counts
        elif (df[col].nunique() < 150) and df[col].dtype != 'object' :
            print(df[col].value_counts(bins = 10, dropna=False))
        
        # Space for readability 
        print('')