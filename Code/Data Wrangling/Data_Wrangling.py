# %% [markdown]
# # Data Wrangling

# %% [markdown]
# ## Gather

# %%
# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
import great_tables as gt
import os


# Set default plotting style and set all parematers as standard ready to publish plots
def set_plotting_style():
    # Set the default style
    sns.set_style("darkgrid")

    # Set the default context with font scale
    sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 1.0})

    # Set the default color palette
    sns.set_palette("plasma")

    # Set the default figure size
    plt.rcParams["figure.figsize"] = [10, 6]


# Call the function to set the default plotting style
set_plotting_style()

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# %%
# Define the file path
train_file_path = (
    r"C:/Users/HP/Desktop/Customer Churn Prediction/Data/Raw/churn-bigml-80.csv"
)
test_file_path = (
    r"C:\Users\HP\Desktop\Customer Churn Prediction\Data\Raw\churn-bigml-20.csv"
)

# Read the CSV file into a DataFrame
churn_train = pd.read_csv(train_file_path)
churn_test = pd.read_csv(test_file_path)

# %%
# Display the first 5 rows of the training DataFrame
churn_train.head()

# %%
# Display the first 5 rows of the testing DataFrame
churn_test.head()

# %% [markdown]
# ## Assess

# %% [markdown]
# We assess for potential erroneous records in our data including but not limited to:
# - Missing data
# - Duplicate data
# - Incorrect data types
# - Incorrect data values
#

# %% [markdown]
# ### Missing Data

# %%
# Assess for missing values in the train set
print(churn_train.info())
msno.bar(churn_train)

# %%
# Assess for missing values in the test set
print(churn_test.info())
msno.bar(churn_test)

# %% [markdown]
# ### Duplicates

# %%
# Check for duplicates in both the train and test sets
print(f"The are {churn_train.duplicated().sum()} duplicates in the train set.")
print(f"The are {churn_test.duplicated().sum()} duplicates in the test set.")

# %% [markdown]
# ### Data Types

# %%
# Get the data types of the features of the train set.
churn_train.dtypes

# %%
# Get the data types of the features of the test set
churn_test.dtypes

# %% [markdown]
# Summary of findings:
# - No missing values
# - No duplicates
# - The features have consistent data types with the data they represent.
# - Some features are irrelevant, they do not aid in solving our problem.
# - Some features need to be encoded (will be done in the feature engineering phase)

# %% [markdown]
# ## Clean

# %% [markdown]
# We'll just drop the irrelevant feature at this stage, which is the `State` variable and Clean the feature names.

# %%
# Drop the State variable in our data
churn_train.drop("State", axis=1, inplace=True)
churn_test.drop("State", axis=1, inplace=True)

# %%
# Clean the column names
churn_train.columns = (
    churn_train.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)
churn_test.columns = (
    churn_test.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

# %%
# Display the first 5 rows of our data
churn_train.head()

# %%
# Display the first 5 rows of the test data
churn_test.head()

# %% [markdown]
# ## Test

# %%
# Display the first 5 rows of our data


def show_first_five(data):
    """
    Display the first 5 rows of the given data.

    Parameters:
    data (pandas DataFrame): The data to display

    Returns:
    None
    """

    return data.head()


# %%
show_first_five(churn_train)

# %%
show_first_five(churn_test)

# %%
# Save the train data
churn_train.to_csv(
    "C:/Users/HP/Desktop/Customer Churn Prediction/Data/Processed/churn_train.csv",
    index=False,
)

# Save the test data
churn_test.to_csv(
    "C:/Users/HP/Desktop/Customer Churn Prediction/Data/Processed/churn_test.csv",
    index=False,
)
