import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    
#DECISION TREE

# Split the dataset into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
# Initialize the decision tree classifier
clf = DecisionTreeClassifier(max_depth=8, random_state=42)

# Train the decision tree model
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['Not Churn', 'Churn'], filled=True, rounded=True)
plt.show()