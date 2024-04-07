import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# LOAD DATASET

# Load dataset into a pandas DataFrame
pd.set_option("display.max_columns", None)
df = pd.read_csv('Customer Churn.csv')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# DESCRIPTIVE STATISTICS
numerical_summary = df.describe()
print(numerical_summary)

# Calculate mode for each column
modes = df.mode()
# Display modes
print("Mode of each column:")
print(modes)

# Calculate median for each column
medians = df.median()
# Display medians
print("Median of each column:")
print(medians)

# Calculate mean for each column
means = df.mean()
# Display means
print("Mean of each column:")
print(means)

# Calculate standard deviation for each column
std_devs = df.std()
# Display standard deviations
print("Standard deviation of each column:")
print(std_devs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # GRAPHS

# # BOXPLOTS
# # Set the figure size and style
# plt.figure(figsize=(12, 6))
# sns.set(style="whitegrid")

# # Create the boxplot
# ax = sns.boxplot(data=df.select_dtypes(include='number'))

# # Rotate the x-axis labels
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# # Set title and show plot
# plt.title('Boxplot of Numerical Variables')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.show()


# # HISTOGRAMS
# # Plot histogram for the dataset
# plt.figure(figsize=(12, 8))  # Set the figure size

# for column in df.columns:
#     plt.hist(df[column], bins=20, alpha=0.7, label=column)

# plt.xlabel('Values')  # Set the x-axis label
# plt.ylabel('Frequency')  # Set the y-axis label
# plt.title('Histogram of Each Variable')  # Set the title
# plt.legend()  # Show legend with variable names
# plt.grid(True)  # Add gridlines
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.tight_layout()  # Adjust layout to prevent overlapping
# plt.show()  # Show the plot


# # Plot histogram for each variable in the dataset
# for column in df.columns:
#     plt.figure(figsize=(8, 6))  # Set the figure size
#     plt.hist(df[column], bins=20, color='skyblue', edgecolor='black')  # Plot histogram
#     plt.xlabel('Values')  # Set the x-axis label
#     plt.ylabel('Frequency')  # Set the y-axis label
#     plt.title(f'Histogram of {column}')  # Set the title with variable name
#     plt.grid(True)  # Add gridlines
#     plt.show()  # Show the plot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CLEANING DATA

# Check for missing values
print("Missing values of each column:")
missing_values = df.isnull().sum()
print(missing_values)

# Check for outliers
# Define detect_outliers function to calculate outliers using IQR
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers

# Iterate over each column and calculate outliers using IQR
for column in df.columns:
    outliers = detect_outliers(df[column])
    if not outliers.empty:
        print(f"Potential outliers in {column}:\n{outliers}\n")


#CHECK CATEGORY VALUES & REMOVE ROWS THAT DO NOT MEET THE CONDITION
        
# Complains should be either 0 or 1
# Use boolean indexing to select rows that meet the condition
filtered_df = df[df["Complains"].isin([0, 1])]

print("\nDataFrame after deleting rows based on Complains condition:")
print(filtered_df)
df = pd.DataFrame(filtered_df)

# Age Group should be between 1 and 5
# Use boolean indexing to select rows that meet the condition
filtered_df = df[~(df["Age Group"] > 5)]

print("\nDataFrame after deleting rows based on Age Group condition:")
print(filtered_df)
df = pd.DataFrame(filtered_df)

# Tariff Plan should be either 1 or 2
# Use boolean indexing to select rows that meet the condition
filtered_df = df[df["Tariff Plan"].isin([1, 2])]
print("\nDataFrame after deleting rows based on condition:")
print(filtered_df)
df = pd.DataFrame(filtered_df)

# Status should be either 1 or 2
# Use boolean indexing to select rows that meet the condition
filtered_df = df[df["Status"].isin([1, 2])]
print("\nDataFrame after deleting rows based on Status condition:")
print(filtered_df)
df = pd.DataFrame(filtered_df)


# Churn should be either 0 or 1       
# Use boolean indexing to select rows that meet the condition
filtered_df = df[df["Churn"].isin([0, 1])]

print("\nDataFrame after deleting rows based on Churn condition:")
print(filtered_df)

# Create a new DataFrame with the filtered data
df = pd.DataFrame(filtered_df)     


# Charge Amount should be between 0 and 9        
# Use boolean indexing to select rows that meet the condition
filtered_df = df[~(df['Charge  Amount'] > 9)]

print("\nDataFrame after deleting rows based on Charge Amount condition:")
print(filtered_df)

# Create a new DataFrame with the filtered data
df = pd.DataFrame(filtered_df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#CORRELATION ANALYSIS

#Chi-Square Test for Categorical Columns
# Extract the columns
from scipy.stats import chi2_contingency

# List of categorical columns
categorical_cols = ['Complains', 'Charge  Amount', 'Status', 'Tariff Plan', 'Age Group']

# Perform Chi-Square test for each categorical column
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df['Churn'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square test for {col} and Churn:")
    print(f"Chi2 statistic: {chi2}")
    print(f"p-value: {p}")
    print(f"Degrees of freedom: {dof}")
    print(f"Expected contingency table: {expected}\n")

print(df.columns)


#Pearson Correlation for Numeric Columns
# List of numeric columns
numeric_cols = ['Call  Failure', 'Subscription  Length', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Age', 'Customer Value']

# Create a new DataFrame that only includes the numeric columns
numeric_df = df[numeric_cols]

# Compute and print Pearson correlation coefficient for each numeric column and 'Churn'
for col in numeric_cols:
    correlation = df[col].corr(df['Churn'], method='pearson')
    print(f"Pearson correlation between {col} and Churn: {correlation}")

#Results are extremely low, since the data is skewed and not linearly related it might be inappropriate to use Pearson correlation without transforming the data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# DECISION TREE

# this node object represents the node in the tree and keeps track of
# the predicted class (majority class) in the case of a leaf node
# the feature and threshold it splitted on
# the left and right child nodes (sub trees)
# initially the variables have no values until the tree is built with all it's nodes
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

# Calculates the Gini impurity of the input array y. This represents the probability
# of misclassifying a randomly chosen element in y if it were randomly labeled
# according to the class distribution in y.
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    proba = counts / len(y)
    return 1 - np.sum(proba ** 2)

# return the subset of the X dataset of independant variables 
#and the subset of the y dataset of the dependant variable split further into left or right
# depending on if they meet the threshold condition for left and right
def split(X, y, index, threshold):
    left = np.where(X[:, index] < threshold)
    right = np.where(X[:, index] >= threshold)
    return X[left], X[right], y[left], y[right]

# returns the best feature/column's index and the best threshold value to split the node at
# loops through all columns in the X dataset of independant variables and checks all unique values as candiate thresholds
# whichever one has the best gini index becomes the feature and threshold for the node and therefore returns them
def best_split(X, y):
    best_index, best_threshold, best_gini = 0, 0, 1
    for column_index in range(X.shape[1]):
        thresholds = np.unique(X[:, column_index])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split(X, y, column_index, threshold)
            gini_left, gini_right = gini(y_left), gini(y_right)
            gini_total = len(y_left) / len(y) * gini_left + len(y_right) / len(y) * gini_right
            if gini_total < best_gini:
                best_gini = gini_total
                best_index = column_index
                best_threshold = threshold
    return best_index, best_threshold

# Recursive function to build the decision tree that takes
# a set of data that contains the independent variables (X)
# the target variable (y)
# the current depth of the tree (depth)
# and the maximum depth of the tree (max_depth)
def build_tree(X, y, depth=0, max_depth=5):
    # find the unique class values for the dependant variable (churn) and return the counts for each class and the unique classes
    classes, counts = np.unique(y, return_counts=True)
    # If y is empty, return a node with no predicted class
    if len(counts) == 0:
        return Node(None)
    # store the index of the class with the most counts in the predicted class and pass it to the new node object 
    predicted_class = classes[np.argmax(counts)]
    node = Node(predicted_class=predicted_class)
    # check for max depth in each recursive call and stop building the tree when reached
    if depth < max_depth:
        index, threshold = best_split(X, y) # returns the best column/feature's index and threshold to split the node at
        # check for a valid split
        if index is not None:
            # splits the node and stores the feature and threshold the node was split at as well as creates the left and right child nodes (sub trees)
            X_left, X_right, y_left, y_right = split(X, y, index, threshold)
            node.feature_index = index
            node.threshold = threshold
            node.left = build_tree(X_left, y_left, depth + 1, max_depth)
            node.right = build_tree(X_right, y_right, depth + 1, max_depth)
    return node

# predicts the class for each instance of x/sample in the test dataset
def predict(node, X):
    # Leaf node so it is pure enough and does not split further, return the predicted class
    if node.left is None:
        return node.predicted_class
    # else check whether to go to the left child node or right child node to make a decision
    if X[node.feature_index] < node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)

# Split the dataset into features (X) and target (y)
X = df.drop('Churn', axis=1).values
y = df['Churn'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the decision tree and limit to depth of 8 for optimal performance
tree = build_tree(X_train, y_train, max_depth=8)

# return the predictions for the test dataset in an array
predictions = [predict(tree, x) for x in X_test]

# takes the average of the predictions that match the actual values in the test dataset to calculate the accuracy
accuracy = (predictions == y_test).mean()
print(f'Accuracy: {accuracy}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Split the dataset into features (X) and target (y)
# X = df.drop('Churn', axis=1)
# y = df['Churn']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Model Training
# # Initialize the decision tree classifier
# clf = DecisionTreeClassifier(max_depth=8, random_state=42)

# # Train the decision tree model
# clf.fit(X_train, y_train)

# # Make predictions on the testing set
# y_pred = clf.predict(X_test)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Visualize the decision tree
# plt.figure(figsize=(20,10))
# plot_tree(clf, feature_names=X.columns, class_names=['Not Churn', 'Churn'], filled=True, rounded=True)
# plt.show()


#-------------------------------------------------------------------------------------------
# Logistic Regression Model

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class LogisticRegression():
    # learning rate is lr and number of iterations is n_iters and their respective values.
    def __init__(self, lr=0.0015, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost_list = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
       
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            
            # Gradient Descent ( find maximum limit)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

            # Keep track of our cost function value
            # cost function
            cost = -(1/n_samples)*np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            self.cost_list.append(cost)

            if((_%(self.n_iters)/10) == 0):
                print("cost after",_, "iteration is: ", cost)


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

#--------------------------------------------------------------------------------
Training of the Logistic Regression Model

X, y = df.drop('Churn', axis=1).values, df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return (np.sum(y_pred==y_test)/len(y_test))*100

acc = accuracy(y_pred, y_test)
print("Accuracy of the logistic regression model is = ", round(acc,2), "%")
