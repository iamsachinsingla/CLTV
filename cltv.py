ReadMe
- The project directory is: 'C:/Users/huang/Downloads/HUB-Group2/'
- Use the original data sets, 'Policy Data.xlsx' and 'Lead Data.xlsx'
- Contents
- Part 1 - Data Cleaning for Visualization and Business Understanding
- Part 2 - Data Cleaning for Modeling
- Part 3 - Model Training
- Part 4 - Implementation

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Part 1 - Data Cleaning for Visualization

Import related packages and set directory path
# Import Pandas and Numpy
import pandas as pd
import numpy as np
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#!pip install lightgbm
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

Read policy data as policydf
# Read policy data as policydf
project_dir = 'C:/Users/huang/Downloads/HUB-Group2/'
policydf = pd.read_excel(project_dir+'Policy Data.xlsx', sheet_name='policy')

Map each postal into Province
# Create your mapping dictionary
state_dict = {
    'A': 'NL',
    'B': 'NS',
    'C': 'PE',
    'E': 'NB',
    'G': 'QC',
    'H': 'QC',
    'J': 'QC',
    'K': 'ON',
    'L': 'ON',
    'M': 'ON',
    'N': 'ON',
    'P': 'ON',
    'T': 'AB',
    'V': 'BC',
}
# Create a new column 'postal_initial' by taking the initial letter of the postal code
policydf['postal_initial'] = policydf['Postal'].str[0]
# Create a new column 'Province' by mapping the 'postal_initial' column using the dictionary
policydf['Province'] = policydf['postal_initial'].map(state_dict)
# Drop the helper columns
policydf.drop(columns=['postal_initial'], inplace=True)

Convert three date columns into datetime
# Convert Fwritten, Expire, and Effective to datetime
policydf['Fwritten'] = pd.to_datetime(policydf['Fwritten'])
policydf['Effective'] = pd.to_datetime(policydf['Effective'])
policydf['Expires'] = pd.to_datetime(policydf['Expires'])

Add a column named "Real_Fwritten" to calculate the real file written date
# Add a column named "Real_Fwritten" to calculate the real file written date
def get_date(row):
    if row['Fwritten'].day < row['Effective'].day:
        try:
            # Try to replace the month and day
            date = row['Fwritten'].replace(month=row['Effective'].month, day=row['Effective'].day)
        except ValueError:
            # If day is out of range for month, get last day of month
            date = pd.Timestamp(year=row['Fwritten'].year, month=row['Effective'].month + 1, day=1) - pd.Timedelta(days=1)
        return min(row['Effective'], date)
    else:
        return min(row['Effective'], row['Fwritten'])
policydf['Real_Fwritten'] = policydf.apply(get_date, axis=1)

Set a cut off date, all the policies would be expired after 2023-04-04
# Set cutoff date
cutoff_date = pd.to_datetime('2023-04-04')
# Function to apply
def get_expire_date(row):
    if row['Effective'] > row['Expires']:
        return row['Effective']
    elif row['Expires'] > cutoff_date:
        return cutoff_date
    else:
        return row['Expires']
# Apply the function to create the new column
policydf['2023-04-04Expire'] = policydf.apply(get_expire_date, axis=1)

Add a column named "Age" to calculate the customer's birth year
def get_birth_year(row):
    if pd.isnull(row['Age_FirstWritten']):
        return np.nan
    else:
        return row['Real_Fwritten'].year - row['Age_FirstWritten']
# Apply the function to create the new column
policydf['Birth_Year'] = policydf.apply(get_birth_year, axis=1)
# Get the current year
current_year = datetime.now().year
# Subtract the birth year from the current year to get age
policydf['Age'] = current_year - policydf['Birth_Year']
policydf.drop(columns=['Birth_Year'], inplace=True)
# Check if the age of a single customer with multiple policies are the same, if not, use the first record
policydf['Age'] = policydf.groupby('Cust_ID')['Age'].transform('first')

Add a column named "Is_Active" to check if a policy is active or not
# Set the comparison date
comparison_date = pd.to_datetime('2023-04-04')
# Create 'Is_Active' column
policydf['Is_Active'] = policydf['Expires'].apply(lambda x: 'Active' if x > comparison_date else 'Inactive')

Add a column named 'Cross_Selling' to check if the customer has multiple policies and three columns to count how many auto/property policies and total number of policy a customer has
# Add a column named 'Cross_Selling' to check if the customer has multiple policies
policydf['Cross_Selling'] = policydf['Cust_ID'].duplicated(keep=False)
# Define a function to return 1 if LOB is Auto or Property, 0 otherwise
def check_lob(row, lob_type):
    if row == lob_type:
        return 1
    else:
        return 0
# Apply check_lob to 'LOB' column to create two new columns
policydf['IsAuto'] = policydf['LOB'].apply(lambda x: check_lob(x, 'Auto'))
policydf['IsProperty'] = policydf['LOB'].apply(lambda x: check_lob(x, 'Property'))
# Add a column name 'NumOfAutoPolicy' to count how many auto policies a customer has
policydf['NumOfAutoPolicy'] = policydf.groupby('Cust_ID')['IsAuto'].transform('sum')
# Add a column name 'NumOfPropertyPolicy' to count how many property policies a customer has
policydf['NumOfPropertyPolicy'] = policydf.groupby('Cust_ID')['IsProperty'].transform('sum')
# Drop the helper columns
policydf.drop(columns=['IsAuto', 'IsProperty'], inplace=True)
# Add a column named "TotalNumOfPolicy" to calculate how many policies a customer has
policydf['TotalNumOfPolicy'] = policydf.groupby('Cust_ID')['Cust_ID'].transform('count')

Add a column named 'Retention' to calculate the retention of a customer
# Add a column named 'Retention' to calculate the retention of a customer
policydf['Retention'] = (policydf['2023-04-04Expire'] - policydf['Real_Fwritten']).dt.days / 365
# Check if the retention of a single customer with multiple policies are the same, if not, use the larger one
policydf['Retention'] = policydf.groupby('Cust_ID')['Retention'].transform('max')

Add three columns to calculate the real premium in each year with the consideration of retention
# Add three columns to calculate the real premium in each year with the consideration of retention
# 1stPremium
policydf['1stPremium'] = policydf.apply(lambda row: 0 if (row['Retention'] <= 1 and row['Effective'] == row['Expires']) else row['FirstTerm_Premium']*min(row['Retention'], 1), axis=1)
# 2ndPremium
policydf['2ndPremium'] = policydf.apply(lambda row: 0 if row['Retention'] < 1 else row['SecondTerm_Premium']*max(min(row['Retention']-1, 1), 0), axis=1)
# 3rdPremium
policydf['3rdPremium'] = policydf.apply(lambda row: 0 if row['Retention'] < 2 else row['ThirdTerm_Premium']*max(min(row['Retention']-2, 1), 0), axis=1)

Add six columns to calculate the premium after the third year
# Add six columns to calculate the premium after the third year
# Set initial premium and multiplier
premium = '3rdPremium'
multiplier = 1.05

# For each term from 4th to 9th
for i in range(4, 10):
    # Create new column name
    new_column = '4thPremium' if i == 4 else f'{i}thPremium'

    # Calculate premium
    policydf[new_column] = policydf.apply(lambda row: 0 if row['Retention'] < i-1 else row[premium]*max(min(row['Retention']-(i-1), 1), 0)*multiplier, axis=1)

    # Update multiplier for next term
    multiplier *= 1.05

Add a column named "PolicyLTV" to get the lifetime value of a customer in each policy by summing the premium amount of all 9 years together
# Select columns for summing
columns_to_sum = ['1stPremium', '2ndPremium', '3rdPremium', '4thPremium', '5thPremium', '6thPremium', '7thPremium', '8thPremium', '9thPremium']
# Add 'PolicyLTV' column
policydf['PolicyLTV'] = policydf[columns_to_sum].sum(axis=1)

Add a column named "AutoLTV" to calculate the auto insurance lifetime value of a single customer
# Add a column named "AutoLTV" to calculate the auto insurance lifetime value of a single customer
policydf['AutoLTV'] = policydf['Cust_ID'].map(policydf[policydf['LOB'] == 'Auto'].groupby('Cust_ID')['PolicyLTV'].sum())
policydf['AutoLTV'] = policydf['AutoLTV'].fillna(0)

Add a column named "PropertyLTV" to calculate the property insurance lifetime value of a single customer
# Add a column named "PropertyLTV" to calculate the auto insurance lifetime value of a single customer
policydf['PropertyLTV'] = policydf['Cust_ID'].map(policydf[policydf['LOB'] == 'Property'].groupby('Cust_ID')['PolicyLTV'].sum())
policydf['PropertyLTV'] = policydf['PropertyLTV'].fillna(0)

Add a column named "CustomerLTV" to calculate the total customer lifetime value of a single customer
# Add a column named "CustomerLTV" to calculate the total customer lifetime value of a single customer
policydf['CustomerLTV'] = policydf.groupby('Cust_ID')['PolicyLTV'].transform('sum')

Sort rows based on CustomerLTV and LOB in ascending order, then keep the first record for each customer
# Sort the dataframe in a descending order based on CustomerLTV and LOB
policydf = policydf.sort_values(by=['CustomerLTV', 'LOB'], ascending=[False, True])
# Keep the first record for each customer
policydf = policydf.drop_duplicates(subset='Cust_ID', keep='first')

Some other data cleaning process
# Replace X to null in column "Gender"
policydf.loc[:, 'Gender'] = policydf['Gender'].replace('X', np.nan)
# Replace all the values except for "M", "S", and null into "Others"
policydf['Maritalsta'] = policydf['Maritalsta'].replace(['C', 'D', 'N', 'P', 'R', 'W', 'Y'], 'Others')
# Delete the rows where Years_Licensed_at_FW is less than 0
policydf = policydf[policydf['Years_Licensed_at_FW'] >= 0]
# Delete the rows where FirstTerm_Premium is less than 0
policydf = policydf[policydf['FirstTerm_Premium'] >= 0]
# Delete rows with null values in related columns
#columns_to_dropna = ['FirstTerm_Premium', 'Driver_Count', 'Vehicle_Count', 'Years_Licensed_at_FW', 'Age', 'Gender', 'Maritalsta','Province']
#policydf = policydf.dropna(subset=columns_to_dropna)

Label the top 20% customers as ClassA, the rest as Others
# Define the cutoff for the top 20%
cutoff = policydf['CustomerLTV'].quantile(0.8)
# Use numpy where to assign labels based on the cutoff and retention
policydf['Class'] = np.where(policydf['CustomerLTV'] > cutoff, 'ClassA', 'Others')

Export the cleaned dataset
policydf.to_csv('C:/Users/huang/Downloads/HUB-Group2/policydf.csv', index=False)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Part 2 - Data Cleaning for Modeling

Read policy data as policydf
# Read policy data as policydf
project_dir = 'C:/Users/huang/Downloads/HUB-Group2/'
policydf = pd.read_excel(project_dir+'Policy Data.xlsx', sheet_name='policy')

Map each postal into Province
# Create your mapping dictionary
state_dict = {
    'A': 'NL',
    'B': 'NS',
    'C': 'PE',
    'E': 'NB',
    'G': 'QC',
    'H': 'QC',
    'J': 'QC',
    'K': 'ON',
    'L': 'ON',
    'M': 'ON',
    'N': 'ON',
    'P': 'ON',
    'T': 'AB',
    'V': 'BC',
}
# Create a new column 'postal_initial' by taking the initial letter of the postal code
policydf['postal_initial'] = policydf['Postal'].str[0]
# Create a new column 'Province' by mapping the 'postal_initial' column using the dictionary
policydf['Province'] = policydf['postal_initial'].map(state_dict)
# Drop the helper columns
policydf.drop(columns=['postal_initial'], inplace=True)

Convert Fwritten, Expire, and Effective to datetime
# Convert Fwritten, Expire, and Effective to datetime
policydf['Fwritten'] = pd.to_datetime(policydf['Fwritten'])
policydf['Effective'] = pd.to_datetime(policydf['Effective'])
policydf['Expires'] = pd.to_datetime(policydf['Expires'])

Add a column named "Real_Fwritten" to calculate the real file written date
# Add a column named "Real_Fwritten" to calculate the real file written date
def get_date(row):
    if row['Fwritten'].day < row['Effective'].day:
        try:
            # Try to replace the month and day
            date = row['Fwritten'].replace(month=row['Effective'].month, day=row['Effective'].day)
        except ValueError:
            # If day is out of range for month, get last day of month
            date = pd.Timestamp(year=row['Fwritten'].year, month=row['Effective'].month + 1, day=1) - pd.Timedelta(days=1)
        return min(row['Effective'], date)
    else:
        return min(row['Effective'], row['Fwritten'])
policydf['Real_Fwritten'] = policydf.apply(get_date, axis=1)

Add a column named "Age" to calculate the customer's birth year
# Add a column named "Age" to calculate the customer's birth year
def get_birth_year(row):
    if pd.isnull(row['Age_FirstWritten']):
        return np.nan
    else:
        return row['Real_Fwritten'].year - row['Age_FirstWritten']
# Apply the function to create the new column
policydf['Birth_Year'] = policydf.apply(get_birth_year, axis=1)
# Get the current year
current_year = datetime.now().year
# Subtract the birth year from the current year to get age
policydf['Age'] = current_year - policydf['Birth_Year']
policydf.drop(columns=['Birth_Year'], inplace=True)
# Check if the age of a single customer with multiple policies are the same, if not, use the first record
policydf['Age'] = policydf.groupby('Cust_ID')['Age'].transform('first')

Add a column named 'Retention' to calculate the retention of a customer
# Add a column named 'Retention' to calculate the retention of a customer
policydf['Retention'] = (policydf['Expires'] - policydf['Real_Fwritten']).dt.days / 365
# Check if the retention of a single customer with multiple policies are the same, if not, use the larger one
policydf['Retention'] = policydf.groupby('Cust_ID')['Retention'].transform('max')

Add three columns to calculate the real premium in each year with the consideration of retention
# Add three columns to calculate the real premium in each year with the consideration of retention
# 1stPremium
policydf['1stPremium'] = policydf.apply(lambda row: 0 if (row['Retention'] <= 1 and row['Effective'] == row['Expires']) else row['FirstTerm_Premium']*min(row['Retention'], 1), axis=1)
# 2ndPremium
policydf['2ndPremium'] = policydf.apply(lambda row: 0 if row['Retention'] < 1 else row['SecondTerm_Premium']*max(min(row['Retention']-1, 1), 0), axis=1)
# 3rdPremium
policydf['3rdPremium'] = policydf.apply(lambda row: 0 if row['Retention'] < 2 else row['ThirdTerm_Premium']*max(min(row['Retention']-2, 1), 0), axis=1)

Add a column named "CustomerLTV" to calculate the total customer lifetime value of a single customer
# Select columns for summing
columns_to_sum = ['1stPremium', '2ndPremium', '3rdPremium']
# Add 'PolicyLTV' column
policydf['PolicyLTV'] = policydf[columns_to_sum].sum(axis=1)
# Add a column named "CustomerLTV" to calculate the total customer lifetime value of a single customer
policydf['CustomerLTV'] = policydf.groupby('Cust_ID')['PolicyLTV'].transform('sum')

Sort rows based on CustomerLTV and LOB in ascending order, then keep the first record for each customer
# Sort the dataframe in a descending order based on CustomerLTV and LOB
policydf = policydf.sort_values(by=['CustomerLTV', 'LOB'], ascending=[False, True])
# Keep the first record for each customer
policydf = policydf.drop_duplicates(subset='Cust_ID', keep='first')

Drop the customers that have zero LTV
policydf = policydf[policydf['CustomerLTV'] > 0]

Some other data cleaning process
# Replace X to null in column "Gender"
policydf.loc[:, 'Gender'] = policydf['Gender'].replace('X', np.nan)
# Replace all the values except for "M", "S", and null into "Others"
policydf['Maritalsta'] = policydf['Maritalsta'].replace(['C', 'D', 'N', 'P', 'R', 'W', 'Y'], 'Others')
# Delete the rows where Years_Licensed_at_FW is less than 0
policydf = policydf[policydf['Years_Licensed_at_FW'] >= 0]
# Delete the rows where FirstTerm_Premium is less than 0
policydf = policydf[policydf['FirstTerm_Premium'] >= 0]
# Delete rows with null values in related columns
columns_to_dropna = ['FirstTerm_Premium', 'Driver_Count', 'Vehicle_Count', 'Years_Licensed_at_FW', 'Age', 'Gender', 'Maritalsta','Province']
policydf = policydf.dropna(subset=columns_to_dropna)

Label the top 20% customers as Top20 the rest as Others in column "Top20"
Label the bottom 20% customers as Bottom20 the rest as Others in column "Bottom20"
# Define the cutoff for the top 20%
cutoff = policydf['CustomerLTV'].quantile(0.8)
# Use numpy where to assign labels based on the cutoff and retention
policydf['Top20'] = np.where(policydf['CustomerLTV'] > cutoff, 'Top20', 'Others')
# Define the cutoff for the bottom 20%
cutoff = policydf['CustomerLTV'].quantile(0.2)
# Use numpy where to assign labels based on the cutoff and retention
policydf['Bottom20'] = np.where(policydf['CustomerLTV'] > cutoff, 'Others', 'Bottom20')

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Part 3 - Model Training

Count how many records in each category
class_count_top20 = policydf['Top20'].value_counts()
print(class_count_top20)
print('----------------------------')
class_count_bottom_20 = policydf['Bottom20'].value_counts()
print(class_count_bottom_20)

Keep all the minority class and random sample the same amount of majority class
df_top20 = policydf.groupby('Top20').apply(lambda x: x.sample(n=class_count_top20[1], replace=False))
df_top20 = df_top20.reset_index(drop=True)
df_bottom20 = policydf.groupby('Bottom20').apply(lambda x: x.sample(n=class_count_bottom_20[1], replace=False))
df_bottom20 = df_bottom20.reset_index(drop=True)

Double Check if two classes have the same amount of records
class_count_top20 = df_top20['Top20'].value_counts()
print(class_count_top20)
print('----------------------------')
class_count_bottom_20 = df_bottom20['Bottom20'].value_counts()
print(class_count_bottom_20)

Change the class label into 0 and 1, Top20 is 0, Other is 1
Change the class label into 0 and 1, Bottom20 is 0, Other is 1
# Define a dictionary where keys are the current values and values are the new values
class_mapping = {'Top20': 0, 'Others': 1}
# Apply the mapping to the 'Class' column
df_top20['Top20'] = df_top20['Top20'].map(class_mapping)

# Define a dictionary where keys are the current values and values are the new values
class_mapping = {'Bottom20': 0, 'Others': 1}
# Apply the mapping to the 'Class' column
df_bottom20['Bottom20'] = df_bottom20['Bottom20'].map(class_mapping)

Define the dataframe for classification
Xdf_top20 = df_top20[['Age', 'Gender', 'Maritalsta', 'Driver_Count', 'Vehicle_Count', 'Years_Licensed_at_FW', 'Province', 'FirstTerm_Premium']]
ydf_top20 = df_top20['Top20']

Xdf_bottom20 = df_bottom20[['Age', 'Gender', 'Maritalsta', 'Driver_Count', 'Vehicle_Count', 'Years_Licensed_at_FW', 'Province', 'FirstTerm_Premium']]
ydf_bottom20 = df_bottom20['Bottom20']

One-hot encoding for categorical features
# Specify the columns to be encoded and create the transformer
categorical_features = ['Gender', 'Maritalsta', 'Province']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
# Fit the transformer to the dataframe
preprocessor.fit(Xdf_top20)
# Get feature names
feature_names = preprocessor.get_feature_names_out()
# Remove prefixes from feature names
feature_names = [f.split('__')[-1] for f in feature_names]
# Transform the data
Xdf_top20 = preprocessor.transform(Xdf_top20)
# Convert to dataframe
Xdf_top20 = pd.DataFrame(Xdf_top20, columns=feature_names)

# Specify the columns to be encoded and create the transformer
categorical_features = ['Gender', 'Maritalsta', 'Province']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
# Fit the transformer to the dataframe
preprocessor.fit(Xdf_bottom20)
# Get feature names
feature_names = preprocessor.get_feature_names_out()
# Remove prefixes from feature names
feature_names = [f.split('__')[-1] for f in feature_names]
# Transform the data
Xdf_bottom20 = preprocessor.transform(Xdf_bottom20)
# Convert to dataframe
Xdf_bottom20 = pd.DataFrame(Xdf_bottom20, columns=feature_names)

Split the data and train the top20 classifier - by default
Training_Accuracy_top20 = []
Testing_Accuracy_top20 = []
for i in range(10):
  # Split the data into training and testing sets
  X_train_top20, X_test_top20, y_train_top20, y_test_top20 = train_test_split(Xdf_top20, ydf_top20, test_size=0.2)
  # Create a LGBM classifier
  clf_top20 = lgb.LGBMClassifier()
  # Train the classifier
  clf_top20.fit(X_train_top20, y_train_top20)
  # Get predictions for the trainset
  y_pred_train_top20 = clf.predict(X_train_top20)
  train_socre_top20 = accuracy_score(y_train_top20, y_pred_train_top20)
  # Get predictions for the testset
  y_pred_test_top20 = clf_top20.predict(X_test_top20)
  test_score_top20 = accuracy_score(y_test_top20, y_pred_test_top20)
  # Append accuracy score in the list
  Training_Accuracy_top20.append(train_socre_top20)
  Testing_Accuracy_top20.append(test_score_top20)
  if i == 0:
    print("Training Accuracy for top20 in the 1st iteration:", train_socre_top20)
    print("Testing Accuracy for top20 in the 1st iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
  elif i == 1:
    print("Training Accuracy for top20 in the 2nd iteration:", train_socre_top20)
    print("Testing Accuracy for top20 in the 2nd iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
  elif i == 2:
    print("Training Accuracy for top20 in the 3rd iteration:", train_socre_top20)
    print("Testing Accuracy for top20 in the 3rd iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
  else:
    print(f"Training Accuracy for top20 in the {i+1}th iteration:", train_socre_top20)
    print(f"Testing Accuracy for top20 in the {i+1}th iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
print('The average Traning Accuracy top20 is: ', np.mean(Training_Accuracy_top20))
print('The average Testing Accuracy top20 is: ', np.mean(Testing_Accuracy_top20))

Try improve the top20 classifier - by grid search
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [5, 10, 15, 20, 25],
    'num_leaves': [20, 30, 40, 50, 60],
}

# Create a LightGBM classifier
clf_top20 = lgb.LGBMClassifier()

# Create the grid search object
grid_top20 = GridSearchCV(clf_top20, param_grid, cv=5, scoring='accuracy')

# Fit the grid search object to the data
grid_top20.fit(X_train_top20, y_train_top20)

# Print the best parameters and the corresponding score
print(f"Best parameters for top20 classifier is: {grid_top20.best_params_}")
print(f"Best accuracy score for top20 classifier is: {grid_top20.best_score_}")

# Evaluate the best model on the test data
best_clf_top20 = grid_top20.best_estimator_
y_pred_top20 = best_clf_top20.predict(X_test_top20)

print('Testing accuracy for top20 classifier is:', accuracy_score(y_test_top20, y_pred_top20))

Try improve the top20 classifier - by randomized search
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [5, 10, 15, 20, 25],
    'num_leaves': [20, 30, 40, 50, 60],
}

# Create a LightGBM classifier
clf_top20 = lgb.LGBMClassifier()

# Create the random search object
rand_search_top20 = RandomizedSearchCV(clf_top20, param_grid, n_iter=60, cv=5, scoring='accuracy', random_state=42)

# Fit the random search object to the data
rand_search_top20.fit(X_train_top20, y_train_top20)

# Print the best parameters and the corresponding score
print(f"Best parameters for top20 classifier is: {rand_search_top20.best_params_}")
print(f"Best accuracy score for top20 classifier is: {rand_search_top20.best_score_}")

# Evaluate the best model on the test data
best_clf_top20 = rand_search_top20.best_estimator_
y_pred_top20 = best_clf_top20.predict(X_test_top20)

print('Testing accuracy for top20 classifier is:', accuracy_score(y_test_top20, y_pred_top20))

Try improve the top20 classifier - by defined randomized search
# Generate parameter lists
learning_rate_list = [i/100 for i in range(1, 31)]
n_estimators_list = [i for i in range(50, 801)]
max_depth_list = [i for i in range(1, 21)]
num_leaves_list = [i for i in range(2, 51)]

parameter_result_top20 = {}
for i in range(51):
  learning_rate = random.choice(learning_rate_list)
  n_estimators = random.choice(n_estimators_list)
  max_depth = random.choice(max_depth_list)
  num_leaves = random.choice(num_leaves_list)
  # Create a LGBM classifier
  clf_improved_top20 = lgb.LGBMClassifier(learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, num_leaves = num_leaves, force_row_wise = True)
  # Train the classifier
  clf_improved_top20.fit(X_train_top20, y_train_top20)
  # Get predictions for the trainset
  y_pred_train_top20 = clf_improved_top20.predict(X_train_top20)
  train_socre_top20 = accuracy_score(y_train_top20, y_pred_train_top20)
  # Get predictions for the testset
  y_pred_test_top20 = clf_improved_top20.predict(X_test_top20)
  test_score_top20 = accuracy_score(y_test_top20, y_pred_test_top20)
  print(f"Iteration No. {i+1}")
  print('learning_rate is {}, n_estimators is {}, max_depth is {}, num_leaves is {}'.format(learning_rate, n_estimators, max_depth, num_leaves))
  print("Training Accuracy for top20 classifier is:", train_socre_top20)
  print("Testing Accuracy for top20 classifier isis:", test_score_top20)
  print('------------------------------------------------------')
  parameter_result_top20[i+1] =  [learning_rate, n_estimators, max_depth, num_leaves, train_socre_top20, test_score_top20]
  total_iteration = i

# Find the key with the maximum accuracy value
max_key = max(parameter_result_top20, key=lambda k: parameter_result_top20[k][5])
# Print the iteration and the corresponding values
print("\n")
print(f"After {total_iteration} iteration, the highest accuracy value for top20 classifier is iteration {max_key}, and the parameters are listed below:")
print(f"The learning_rate is {parameter_result_top20[max_key][0]}.")
print(f"The n_estimators is {parameter_result_top20[max_key][1]}.")
print(f"The max_depth is {parameter_result_top20[max_key][2]}.")
print(f"The num_leaves is {parameter_result_top20[max_key][3]}.")
print(f"The train_score is {parameter_result_top20[max_key][4]}.")
print(f"The test_score is {parameter_result_top20[max_key][5]}.")

Use the best parameters to train the top20 classifier

￼
Training_Accuracy_top20 = []
Testing_Accuracy_top20 = []
for i in range(10):
  # Split the data into training and testing sets
  X_train_top20, X_test_top20, y_train_top20, y_test_top20 = train_test_split(Xdf_top20, ydf_top20, test_size=0.2)
  # Create a LGBM classifier
  clf_top20 = lgb.LGBMClassifier(learning_rate = 0.09, n_estimators = 700, max_depth = 7, num_leaves = 10, force_row_wise = True)
  # Train the classifier
  clf_top20.fit(X_train_top20, y_train_top20)
  # Get predictions for the trainset
  y_pred_train_top20 = clf_top20.predict(X_train_top20)
  train_socre_top20 = accuracy_score(y_train_top20, y_pred_train_top20)
  # Get predictions for the testset
  y_pred_test_top20 = clf_top20.predict(X_test_top20)
  test_score_top20 = accuracy_score(y_test_top20, y_pred_test_top20)
  # Append accuracy score in the list
  Training_Accuracy_top20.append(train_socre_top20)
  Testing_Accuracy_top20.append(test_score_top20)
  if i == 0:
    print("Training Accuracy for the top20 classifier in 1st iteration:", train_socre_top20)
    print("Testing Accuracy for the top20 classifier in 1st iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
  elif i == 1:
    print("Training Accuracy for the top20 classifier in 2nd iteration:", train_socre_top20)
    print("Testing Accuracy for the top20 classifier in 2nd iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
  elif i == 2:
    print("Training Accuracy for the top20 classifier in 3rd iteration:", train_socre_top20)
    print("Testing Accuracy for the top20 classifier in 3rd iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
  else:
    print(f"Training Accuracy for the top20 classifier in {i+1}th iteration:", train_socre_top20)
    print(f"Testing Accuracy for the top20 classifier in {i+1}th iteration:", test_score_top20)
    print('-------------------------------------------------------------------------------')
print('The average Traning Accuracy top20 classifier is: ', np.mean(Training_Accuracy_top20))
print('The average Testing Accuracy top20 classifier is: ', np.mean(Testing_Accuracy_top20))

Use the best parameters to train the bottom20 classifier
Training_Accuracy_bottom20 = []
Testing_Accuracy_bottom20 = []
for i in range(10):
  # Split the data into training and testing sets
  X_train_bottom20, X_test_bottom20, y_train_bottom20, y_test_bottom20 = train_test_split(Xdf_bottom20, ydf_bottom20, test_size=0.2)
  # Create a LGBM classifier
  clf_bottom20 = lgb.LGBMClassifier(learning_rate = 0.09, n_estimators = 700, max_depth = 7, num_leaves = 10, force_row_wise = True)
  # Train the classifier
  clf_bottom20.fit(X_train_bottom20, y_train_bottom20)
  # Get predictions for the trainset
  y_pred_train_bottom20 = clf_bottom20.predict(X_train_bottom20)
  train_socre_bottom20 = accuracy_score(y_train_bottom20, y_pred_train_bottom20)
  # Get predictions for the testset
  y_pred_test_bottom20 = clf_bottom20.predict(X_test_bottom20)
  test_score_bottom20 = accuracy_score(y_test_bottom20, y_pred_test_bottom20)
  # Append accuracy score in the list
  Training_Accuracy_bottom20.append(train_socre_bottom20)
  Testing_Accuracy_bottom20.append(test_score_bottom20)
  if i == 0:
    print("Training Accuracy for the bottom20 classifier in 1st iteration:", train_socre_bottom20)
    print("Testing Accuracy for the bottom20 classifier in 1st iteration:", test_score_bottom20)
    print('-------------------------------------------------------------------------------')
  elif i == 1:
    print("Training Accuracy for the bottom20 classifier in 2nd iteration:", train_socre_bottom20)
    print("Testing Accuracy for the bottom20 classifier in 2nd iteration:", test_score_bottom20)
    print('-------------------------------------------------------------------------------')
  elif i == 2:
    print("Training Accuracy for the bottom20 classifier in 3rd iteration:", train_socre_bottom20)
    print("Testing Accuracy for the bottom20 classifier in 3rd iteration:", test_score_bottom20)
    print('-------------------------------------------------------------------------------')
  else:
    print(f"Training Accuracy for the bottom20 classifier in {i+1}th iteration:", train_socre_bottom20)
    print(f"Testing Accuracy for the bottom20 classifier in {i+1}th iteration:", test_score_bottom20)
    print('-------------------------------------------------------------------------------')
print('The average Traning Accuracy bottom20 classifier is: ', np.mean(Training_Accuracy_bottom20))
print('The average Testing Accuracy bottom20 classifier is: ', np.mean(Testing_Accuracy_bottom20))

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Part 4 - Implementation

Data cleaning process
# Read policy data as leaddf
leaddf = pd.read_excel(project_dir+'Lead Data.xlsx', sheet_name='lead')

# Rename the columns
leaddf.columns = ['Quote_Create_Date', 'Postal', 'Gender', 'Maritalsta', 'FirstTerm_Premium', 'Age', 'Sold', 'Years_Licensed_at_FW', 'Age_licensed', 'Driver_Count', 'Vehicle_Count', 'Suspension', 'Cancellation', 'Claims', 'Convictions', 'Min_Continued_Ins']

# Create your mapping dictionary
state_dict = {
    'A': 'NL',
    'B': 'NS',
    'C': 'PE',
    'E': 'NB',
    'G': 'QC',
    'H': 'QC',
    'J': 'QC',
    'K': 'ON',
    'L': 'ON',
    'M': 'ON',
    'N': 'ON',
    'P': 'ON',
    'T': 'AB',
    'V': 'BC',
}
# Create a new column 'postal_initial' by taking the initial letter of the postal code
leaddf['postal_initial'] = leaddf['Postal'].str[0]
# Create a new column 'Province' by mapping the 'postal_initial' column using the dictionary
leaddf['Province'] = leaddf['postal_initial'].map(state_dict)
# Drop the helper columns
leaddf.drop(columns=['postal_initial'], inplace=True)

# Replace all the values except for "M", "S", and null into "Others"
leaddf['Maritalsta'] = leaddf['Maritalsta'].replace(['C', 'D', 'P', 'W'], 'Others')
# Delete the rows where Years_Licensed_at_FW is less than 0
leaddf = leaddf[leaddf['Years_Licensed_at_FW'] >= 0]
# Delete the rows where FirstTerm_Premium is less than 0
leaddf = leaddf[leaddf['FirstTerm_Premium'] >= 0]
# Delete rows with null values in related columns
columns_to_dropna = ['FirstTerm_Premium', 'Driver_Count', 'Vehicle_Count', 'Age', 'Years_Licensed_at_FW', 'Gender', 'Maritalsta','Province']
leaddf = leaddf.dropna(subset=columns_to_dropna)

One-hot encoding
# Compose lead
lead = leaddf[['Age', 'Gender', 'Maritalsta', 'Driver_Count', 'Vehicle_Count', 'Years_Licensed_at_FW', 'Province', 'FirstTerm_Premium']]
# Specify the columns to be encoded and create the transformer
categorical_features = ['Gender', 'Maritalsta', 'Province']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
# Fit the transformer to the dataframe
preprocessor.fit(lead)
# Get feature names
feature_names = preprocessor.get_feature_names_out()
# Remove prefixes from feature names
feature_names = [f.split('__')[-1] for f in feature_names]
# Transform the data
lead = preprocessor.transform(lead)
# Convert to dataframe
lead = pd.DataFrame(lead, columns=feature_names)

Predict based on the improved model
# Get predictions for the lead dataset
y_lead_top20 = clf_top20.predict(lead)
y_lead_bottom20 = clf_bottom20.predict(lead)

Attach the prediction to the lead data set
result = leaddf
result['Top20'] = y_lead_top20
result['Bottom20'] = y_lead_bottom20

Change the class label into Top20 and Others, 0 is Top20, 1 is Others
Change the class label into Bottom20 and Others, 0 is Bottom20, 1 is Others
# Define a dictionary where keys are the current values and values are the new values
class_mapping = {0: 'Top20', 1: 'Others'}
# Apply the mapping to the 'Class' column
result['Top20'] = result['Top20'].map(class_mapping)

# Define a dictionary where keys are the current values and values are the new values
class_mapping = {0: 'Bottom20', 1: 'Others'}
# Apply the mapping to the 'Class' column
result['Bottom20'] = result['Bottom20'].map(class_mapping)

Count the number of each category
print('The number of Top 20% is:', result['Top20'].value_counts()[1])
print('The number of Others is:', result['Top20'].value_counts()[0])
print('-------------------------------')
print('The number of Bottom 20% is:', result['Bottom20'].value_counts()[1])
print('The number of Others is:', result['Bottom20'].value_counts()[0])

Comebine the results together and get the "Prediction" column
conditions = [
    (result['Top20'] == 'Top20') & (result['Bottom20'] == 'Others'),
    (result['Top20'] == 'Others') & (result['Bottom20'] == 'Bottom20'),
    (result['Top20'] == 'Others') & (result['Bottom20'] == 'Others'),
    (result['Top20'] == 'Top20') & (result['Bottom20'] == 'Bottom20')
]
values = ['Top20', 'Bottom20', 'Others', 'Top20']
result['Prediction'] = np.select(conditions, values, default='Unknown')
# Drop the columns named "Top20" and "Bottom20"
result.drop(['Top20', 'Bottom20'], axis=1, inplace=True)
# Sort the result table
result = result.sort_values(by='Prediction', ascending=False)
result
class_count = result['Prediction'].value_counts()
print(class_count)
result.to_csv('C:/Users/huang/Downloads/HUB-Group2/result.csv', index=False)
 