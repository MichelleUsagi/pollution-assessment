# Import libraries
import pandas as pd
import numpy as np

#Load dataset
df = pd.read_csv("Data/pollution_dataset.csv")
df.head(5)
df.shape
df.info()

# Check and handling missing values
df.isnull().sum()
# Drop the unnamed columns
df = df.drop(columns=df.columns[df.columns.str.contains("Unnamed")])
# Confirm the have been dropped
df.isnull().sum()

#DOMAIN VALIDATION 
## Humidity must be between 0 and 100
df = df[df["Humidity"].between(0, 100)]

# Pollution concentrations cannot be negative
pollutant_columns = ["PM2.5", "PM10", "CO", "NO2", "SO2"]
for col in pollutant_columns:
    df = df[df[col] >= 0]

df.shape

# Check for outliers
def remove_outliers_iqr(dataframe, columns):
    for col in columns:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        dataframe = dataframe[
            (dataframe[col] >= lower) &
            (dataframe[col] <= upper)
        ]
    return dataframe
# Identify columns that need outlier removal
cols_to_clean = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
df = remove_outliers_iqr(df, cols_to_clean)

#Shape after outlier removal
df.shape

# FEATURE ENGINEERING
# Total particulate matter
df["total_particles"] = df["PM2.5"] + df["PM10"]

# PM2.5 to PM10 ratio (avoid divide-by-zero)
df["pm_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-6)

# Log-transform skewed pollutant variables
for col in pollutant_columns:
    df[f"log_{col}"] = np.log1p(df[col])


# CREATE TARGET VARIABLE
df["unhealthy"] = (df["PM2.5"] > 35).astype(int)

#Target distribution                                          
print(df["unhealthy"].value_counts())

#Check and save cleaned dataset
df.to_csv("Data/cleaned_pollution_dataset.csv", index=False)
print("Clean dataset saved to:", "Data/cleaned_pollution_dataset")
