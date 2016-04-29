import pandas as pd
import numpy as np
import sys
from patsy import dmatrices, dmatrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

# Supplemental ipython notebook provides further details about the analyses, choice of algorithms for classification/clustering and visualizations.

# Function to initialize dataframes from csv files

def initdf(path='.'):
    
    try:
        # Read all data files into dataframes

        df_messages = pd.read_csv(path + '/messages.csv')

        df_users = pd.read_csv(path + '/users.csv')

        df_user_features = pd.read_csv(path + '/user_features.csv')

        df_test = pd.read_csv(path + '/test.csv')
        
    except:
        print "An unexpected error occurred. Either path is incorrect or file is missing.", sys.exc_info()[0]
        raise

    else:
        return (df_messages, df_users, df_user_features, df_test)

df_messages, df_users, df_user_features, df_test = initdf()

# Utility function that joins two given dataframes and returns the joined dataframe
# This is useful when working on Questions 1 and 2

def get_merged_dataframes(df_users, df_messages):
    
    # Perform outer join on users and messages dataframes to combine user's signup date 
    # with messages received since the signup date
    df_joined_users_messages = pd.merge(df_users, df_messages, on='user.id')
    
    # Filter out rows that contain null values in any of the columns listed below:
    # 'user.id', 'signup.date', 'message.date', 'message.count'
    df_joined_users_messages = df_joined_users_messages.dropna(subset=['user.id','signup.date','message.date',
                                                                       'message.count'], how='any')
    
    # Make sure signup.date and message.date are datetime datatypes and same format
    df_joined_users_messages['signup.date'] = pd.to_datetime(df_joined_users_messages['signup.date'], 
                                                             format='%Y-%m-%d')
    df_joined_users_messages['message.date'] = pd.to_datetime(df_joined_users_messages['message.date'], 
                                                              format='%Y-%m-%d')
    
    return df_joined_users_messages

        
# Question 1: Write a function that finds the total number of messages each user has received since signing up on the client's website. 
# Report the total number of users that have received more than 200 messages.      

# Approach: join data from users and messages datasets, retain messages received since user signed up on client's website, 
# group message count by user.id

# Assumptions: none
    
# Given a threshold count, this function finds the total number of users that have received
# messages over the threshold since they signed up on the client's website
# Function returns the total count of users who received more messages than specified threshold

def message_count_by_user(df_users, df_messages, threshold_num_messages=200):
    
    # Call utility function to join the two dataframes
    df_joined_users_messages = get_merged_dataframes(df_users, df_messages)
    
    # Only keep messages that were received since signing up
    df_joined_users_messages = df_joined_users_messages[df_joined_users_messages['message.date'] >= 
                                                        df_joined_users_messages['signup.date']]
        
    # Get the total message count grouped by user.id
    message_count = df_joined_users_messages.groupby('user.id')['message.count'].sum()
    
    df_message_count = message_count.to_frame()
    df_message_count.rename(columns={'message.count': 'total.message.count'}, inplace=True)
      
    # Total number of users that have received more messages than given threshold 
    return len(df_message_count[df_message_count['total.message.count'] > threshold_num_messages])
    
print 'Question 1: %d' %message_count_by_user(df_users, df_messages)


# Question 2: Our client has told us that there are 5 different types of users that visit their website. 
# Using the features provided for each user in user_features.csv, write a function that identifies the 5 different group of users. 
# Report the mean of the total number of messages each group has received since signing up on the client's website. 

# Approach: PCA for dimensionality reduction and visualization + K-means for clustering

# Assumptions: 
# (1) user_features dataset containing features has many missing values. We impute the missing values using respective 
# feature columnn mean. 
# (2) Standardization is performed using MaxAbsScaler to have scaled values range from -1 to +1 without breaking sparsity 
# Other scaling techniques such as StandardScaler or MinMaxScaler could also be applied if always expect data that is not sparse
# (3) Features dataset was reduced using Principal Component Analysis (PCA) for better visualization. First 3 components
# explained most of the variance (~75%), hence clustering was perfomed using these and also got better performance using this approach
# (4) K-means and Agglomerative hierarchical clustering techniques explored. Finally, K-means was selected as 
# it gave better performance measured using scikit-learn library's silhoutte_score metric (please see accompanying ipython notebook
# for more details)

# This function clusters users into one of the 5 groups (0, 1, 2, 3, 4)
# Based on these user group assignments, it computes the mean of total messages each group received
# Function returns the mean of the total number of messages each group has received since signing up on the client's website. 

def avg_message_count_by_group(df_users, df_messages, df_user_features):
    
    columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
 
    features = df_user_features[list(columns)].values

    # Impute missing values to retain all sample data
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(features)

    # Preprocess dataset and standardize features to have normally distributed data
    # MaxAbsScaler allows scaled features to lie between -1 and +1
    X = MaxAbsScaler().fit_transform(X)

    # Apply PCA decomposition and use first 3 components that explain 75% of variance
    reduced_data = decomposition.PCA(n_components=3).fit_transform(X)
    kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
    
    # Predict which group each user belongs to
    cluster_labels = kmeans.fit_predict(reduced_data)    
    df_user_features['group.id'] = cluster_labels
    
    # Call utility function to join the two dataframes
    df_joined_users_messages = get_merged_dataframes(df_users, df_messages)
    df_joined_users_messages_features = get_merged_dataframes(df_user_features, df_joined_users_messages)
      
    # Only keep messages that were received since signing up
    df_joined_users_messages_features = df_joined_users_messages_features[df_joined_users_messages_features['message.date'] 
                                                                          >= df_joined_users_messages_features['signup.date']]
        
    # Get the average message count grouped by group.id
    avg_message_count = df_joined_users_messages_features.groupby('group.id')['message.count'].mean()
    
    # Return the average message count grouped by user groups and rounded to 2 decimals
    return np.round(avg_message_count.tolist(), decimals=2)
        
print "Question 2: ",
print avg_message_count_by_group(df_users, df_messages, df_user_features)

# Question 3: Our client is interested in building a model that can predict the response variable (given in users.csv) for a user given its features. 
# Write a function that takes as input the user features and outputs the predicted response variable. 
# Recall from Question 2 that there are 5 different types of users that visit the client's website. Report the predicted response for the three users in the test.csv file.

# Approach: train logistic regression model on 80% of dataset for classification of user responses (1 or 0)
# use 'l2' regularization to avoid overfitting with regularization strength = 100

# Assumptions: 
# (1) & (2) same as Question 2
# (3) Different classifiers (Logistic Regression, linear SVC, Naive Bayes, Random Forest) were explored . Finally, logistic regression
# was selected as it gave good performance measured using 10-fold cross validation and mean ROC AUC score in least total running time 
# (please see accompanying ipython notebook for more details)

# Given certain user features try to predict user response (1 or 0)
# Function returns the predicted user response for new users based on user features

def predict_user_response(df_users, df_user_features, df_test):
    
    # Perform outer join on users and user_features dataframes to combine user's response 
    # with features specific to each user
    df_joined_users_messages = pd.merge(df_users, df_user_features, on='user.id')
    
    # Filter out rows that contain null values in any of the columns listed below:
    # 'user.id', 'response'
    df_joined_users_features = df_joined_users_messages.dropna(subset=['user.id','response'], how='any')
    
    # Initialize X and y variables with features and outcome respectively
    y, X = dmatrices('response ~ f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10', 
    df_joined_users_features, return_type="dataframe")
    y = np.ravel(y)
    
    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
    # We use logistic regression classifier as it gives good performance in least running time  
    # Train model using the train dataset, which is 80% of the original dataset
    clf = LogisticRegression(C=100, penalty='l2', tol=0.01)
    clf.fit(X_train,y_train)
    
    # Test the trained model on a new test dataset
    # Test data has missing values for some features. 
    # We fill these missing values using mean of respective columns    
    df_test = df_test.fillna(df_test.mean())
    
    # Initialize test variable with features from the test dataframe
    X_test = dmatrix('f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10', 
                     df_test, return_type="dataframe")

    X_test = MaxAbsScaler().fit_transform(X_test)
     
    # Use the trained logistic regression model to predict user responses on test dataset and return values           
    predicted_response = clf.predict(X_test)
    return [int(response) for response in predicted_response]

print "Question 3: ",
print predict_user_response(df_users, df_user_features, df_test)
         
