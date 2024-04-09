import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
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
print("\nDESCRIPTIVE STATISTICS")
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
print("\nBOXPLOT AND HISTOGRAMS")

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
print("\nDATA CLEANING")

# Check for missing values
print("Missing values of each column:")
missing_values = df.isnull().sum()
print("\n",missing_values)

outliers_count = 0;
# Check for outliers
# Define detect_outliers function to calculate outliers using IQR
def detect_outliers(column):
    global outliers_count
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    outliers_count += len(outliers) # stores total # of outliers across all columns
    return outliers

# List of numeric columns
numeric_cols = ['Call  Failure', 'Subscription  Length', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Age', 'Customer Value']
# Iterate over each column and calculate outliers using IQR
for column in numeric_cols:
    outliers = detect_outliers(df[column])
    if not outliers.empty:
        print(f"Potential outliers in {column}:\n{outliers}\n")

print("\n", outliers_count)

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
print("\nCORRELATION ANALYSIS")

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

#Pearson Correlation for Numeric Columns
# Create a new DataFrame that only includes the numeric columns
numeric_df = df[numeric_cols]

# Compute and print Pearson correlation coefficient for each numeric column and 'Churn'
for col in numeric_cols:
    correlation = df[col].corr(df['Churn'], method='pearson')
    print(f"Pearson correlation between {col} and Churn: {correlation}")

#Results are extremely low, since the data is skewed and not linearly related it might be inappropriate to use Pearson correlation without transforming the data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EVALUTATION METHODS

# Accuracy
def accuracy(y_pred, y_test):
    return (np.sum(y_pred==y_test)/len(y_test))*100

# K-Fold cross validation
def k_fold(X, Y, model, k):
    # shuffle the indices so we take random samples each time
    # only using one set of data (y) so that the indices correspond
    # to the same values for both x and y
    indices = np.random.permutation(len(Y))
    mixedX = X[indices]
    mixedY = Y[indices]

    # split x and y into k subsets
    subsetsX = np.array_split(mixedX, k)
    subsetsY = np.array_split(mixedY, k)

    # pick one subset in the subsets array for validation
    # merge the rest and use for training, do this k times
    scores = []
    for i in range(k):
        # Pick a different subset for validation each loop
        validationX = subsetsX[i]
        validationY = subsetsY[i]

        # Use the rest for training
        trainX = np.concatenate([subset for j, subset in enumerate(subsetsX) if j != i])
        trainY = np.concatenate([subset for j, subset in enumerate(subsetsY) if j != i])

        #train the model
        model.fit(trainX, trainY)
        y_pred = model.predict(validationX)

        # test for accuracy
        acc = accuracy(y_pred, validationY)
        scores.append(acc)

    #calculate average accuray scores and return
    average_score = np.mean(scores)
    return average_score, scores

# T test (compare the models)

def t_test(model1_scores, model2_scores):
    diff = np.array(model1_scores) - np.array(model2_scores)

    # Calculate mean difference between both models
    mean_diff = np.mean(diff)
    
    # Calculate standard error of the mean difference
    standard_error = np.std(diff, ddof=1) / np.sqrt(len(diff))
    
    # Calculate t-statistic
    t_stat = mean_diff / standard_error

    df = len(diff) - 1

    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    alpha = 0.05  # Significance level

    if p_value >= alpha:
        print("There is a significant difference between the performance of the two models")
    else:
        print("There is no significant difference between the performance of the two models")

    return t_stat, p_value

# Confusion Matrix and Precision Metrics Calucation Classes

def custom_confusion_matrix(y_true, y_pred, labels):
    # print(y_true, "\n")
    # print(y_pred, "\n")
    """
    Custom implementation of confusion matrix
    
    Parameters:
    - y_true (array-like): True labels
    - y_pred (array-like): Predicted labels
    - labels (list): List of unique labels in the dataset
    
    Returns:
    - cm (numpy array): Confusion matrix
    """
    num_labels = len(labels)
    cm = np.zeros((num_labels, num_labels), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
        
    return cm

def calculate_metrics(confusion_matrix):
    # Calculates precision, recall, and F1 score from the results of the confusion matrix.

    # Extract values from confusion matrix
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    
    # Calculate precision
    precision = TP / (TP + FP)
    
    # Calculate recall
    recall = TP / (TP + FN)
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

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

    # Add an indented block of code here if needed
    def something():
        return 1
    
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    # Calculates the Gini impurity of the input array y. This represents the probability
    # of misclassifying a randomly chosen element in y if it were randomly labeled
    # according to the class distribution in y.
    def gini_index(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probability = counts / len(y)
        return 1 - np.sum(probability ** 2)

    # return the subset of the X dataset of independant variables 
    # and the subset of the y dataset of the dependant variable split further into left or right
    # depending on if they meet the threshold condition for left and right
    def split_node(self, X, Y, col_index, threshold):
        left_condition = X[:, col_index] < threshold
        right_condition = X[:, col_index] >= threshold
        left = np.where(left_condition)
        right = np.where(right_condition)
        return X[left], X[right], Y[left], Y[right]

    # returns the best feature/column's index and the best threshold value to split the node at
    # loops through all columns in the X dataset of independant variables and checks all unique values as candiate thresholds
    # whichever one has the best gini index becomes the feature and threshold for the node and therefore returns them
    def find_best_split(self,X, Y):
        best_index, best_threshold, best_gini = 0, 0, 1
        for column_index in range(X.shape[1]):
            thresholds = np.unique(X[:, column_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split_node(X, Y, column_index, threshold)
                gini_left, gini_right = self.gini_index(y_left), self.gini_index(y_right)
                gini_total = len(y_left) / len(Y) * gini_left + len(y_right) / len(Y) * gini_right
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
    def build_tree(self, X, Y, depth=0):
        # find the unique class values for the dependant variable (churn) and return the counts for each class and the unique classes
        classes, counts = np.unique(Y, return_counts=True)
        # If y is empty, return a node with no predicted class
        if len(counts) == 0:
            return None
        # store the index of the class with the most counts in the predicted class and pass it to the new node object 
        predicted_class = classes[np.argmax(counts)]
        node = Node(predicted_class=predicted_class)
        # check for max depth in each recursive call and stop building the tree when reached
        if depth < self.max_depth:
            index, threshold = self.find_best_split(X, Y) # returns the best column/feature's index and threshold to split the node at
            # check for a valid split
            if index is not None:
                # splits the node and stores the feature and threshold the node was split at as well as creates the left and right child nodes (sub trees)
                X_left, X_right, y_left, y_right = self.split_node(X, Y, index, threshold)
                node.feature_index = index
                node.threshold = threshold
                node.left = self.build_tree(X_left, y_left, depth + 1)
                node.right = self.build_tree(X_right, y_right, depth + 1)
        return node
    
    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        return np.array([self.predict_node(self.root, x) for x in X])

    # predicts the class for each instance of x/sample in the test dataset
    def predict_node(self, node, X):
        # Leaf node so it is pure enough and does not split further, return the predicted class
        if node.left is None:
            return node.predicted_class
        
        # else check whether to go to the left child node or right child node to make a decision
        if X[node.feature_index] < node.threshold:
            return self.predict_node(node.left, X)
        else:
            return self.predict_node(node.right, X)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training the Decision Tree

print("\nDECISION TREE")

# Split the dataset into features (X) and target (y)
X = df.drop('Churn', axis=1).values
Y = df['Churn'].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Build the decision tree and limit to depth of 8 for optimal performance
tree = DecisionTree(max_depth=8)

#train the model
tree.fit(X_train, Y_train)

# return the predictions for the test dataset in an array
y_pred = tree.predict(X_test)

# takes the average of the predictions that match the actual values in the test dataset to calculate the accuracy
acc = accuracy(y_pred, Y_test)
print(f'Accuracy: {acc}', "%")

k_fold_acc, tree_scores = k_fold(X, Y, tree, 5)
print("k-fold accuracy: " , k_fold_acc, "%")

# Calculating confusion matrix and precision metrics for decision tree algorithm
# Compute confusion matrix
# Ensure y_test and y_pred have the same length
y_test = Y_test[:len(y_pred)]

# Compute confusion matrix
custom_labels = np.unique(np.concatenate((y_test, y_pred)))
custom_cm = custom_confusion_matrix(y_test, y_pred, labels=custom_labels)
print("Custom Confusion Matrix:")
print(custom_cm)

# Compare with scikit-learn's confusion matrix
sklearn_cm = confusion_matrix(y_test, y_pred)
print("\nScikit-learn's Confusion Matrix:")
print(sklearn_cm)

#Caluclate metrics from confusion matrix
precision, recall, f1_score = calculate_metrics(custom_cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Logistic Regression Model

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.015, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_list = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Predictions using sigmoid function ( also known as z
            y_predicted = self.sigmoid(linear_model)

            # calculate cost (error rate) using cost function
            cost = -(1/num_samples)*np.sum(y*np.log(y_predicted) + (1 - y)*np.log(1 - y_predicted))
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.cost_list.append(cost)

            # keep track of our cost fucntion value
           
           
            # print("cost after",_, "iteration is: ", cost)
    
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted = y_predicted > 0.5
        return  y_predicted


#--------------------------------------------------------------------------------
# Training of the Logistic Regression Model
print("\n=LOGISTIC REGRESSION")

# Split the data into training and testing sets
X, y = df.drop('Churn', axis=1).values, df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the custom logistic regression model
model = CustomLogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

acc = accuracy(y_pred, y_test)
print("Accuracy of the logistic regression model is = ", round(acc,2), "%")

k_fold_acc, log_scores = k_fold(X, y, model, 5)
print("k-fold accuracy: " , k_fold_acc, "%")
#--------------------------------------------------------------------------------------
# Random Forest Algorithm
            
class RandomForest:
    def __init__(self, num_trees=10, max_depth=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        for _ in range(self.num_trees):
            # Bootstrap sample for each tree
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Create a decision tree and fit on the bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Append the tree to the forest
            self.trees.append(tree)
    
    def predict(self, X):
        # Make predictions with each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Aggregate predictions using majority voting
        aggregated_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return aggregated_predictions

#--------------------------------------------------------------------------------------
# Random Forest Usage
print("\nRANDOM FOREST")
# Split the dataset into features (X) and target (y)
X = df.drop('Churn', axis=1).values
y = df['Churn'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the random forest
forest = RandomForest(num_trees=10, max_depth=2)

# Train the random forest
forest.fit(X_train, y_train)

# Predict using the random forest
y_pred = forest.predict(X_test)

print(y_pred)

# Calculate accuracy
acc = (y_pred == y_test).mean()
print(f'Accuracy: {acc}')

# K-fold cross validation
k_fold_acc, rf_scores = k_fold(X, y, forest, 5)
print("k-fold accuracy: " , k_fold_acc, "%")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# T-test
print("\nT-test of tree against log")
t_test(tree_scores, log_scores)
print("\nT-test of tree against random forest")
t_test(tree_scores, rf_scores)
print("\nT-test of log against random forest")
t_test(log_scores, rf_scores)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compute confusion matrix
# Ensure y_test and y_pred have the same length
y_test = y_test[:len(y_pred)]

# Compute confusion matrix
custom_labels = np.unique(np.concatenate((y_test, y_pred)))
custom_cm = custom_confusion_matrix(y_test, y_pred, labels=custom_labels)
print("\nCustom Confusion Matrix for Random Forest:")
print(custom_cm)

# Compare with scikit-learn's confusion matrix
sklearn_cm = confusion_matrix(y_test, y_pred)
print("\nScikit-learn's Confusion Matrix for Random Forest:")
print(sklearn_cm)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Caluclate metrics from confusion matrix and precision metrics for forest tree algorithm
precision, recall, f1_score = calculate_metrics(custom_cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
