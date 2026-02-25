# Import libraries
import pandas as pd
import numpy as np

#Load dataset
df = pd.read_csv("Data/pollution_dataset.csv")
df.head()
# Print shape of dataset
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



# Shape after domain validation
df.shape

# FIX DATA TYPES 
for col in pollutant_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows that became NaN after conversion
df = df.dropna(subset=pollutant_columns)

#Shape after data type correction
df.shape


# Remove outliersdef remove_outliers_iqr(dataframe, columns):
def remove_outliers_iqr(dataframe, columns):
    df_clean = dataframe.copy()

    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df_clean = df_clean[
                (df_clean[col] >= lower) &
                (df_clean[col] <= upper)]

    return df_clean


df = remove_outliers_iqr(df, pollutant_columns)

#Shape after outlier removal
df.shape

# FEATURE ENGINEERING
# Total particulate matter
df["total_particles"] = df["PM2.5"] + df["PM10"]

# PM ratio (avoid division by zero properly)
df["pm_ratio"] = df["PM2.5"] / df["PM10"].replace(0, np.nan)

# Replace NaN created by division with 0
df["pm_ratio"] = df["pm_ratio"].fillna(0)

# Log-transform skewed pollutant variables
for col in pollutant_columns:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])


# CREATE TARGET VARIABLE

# Using WHO standard threshold: PM2.5 > 35 µg/m³ → Unhealthy
df["unhealthy"] = (df["PM2.5"] > 35).astype(int)

#Target distribution                                          
print(df["unhealthy"].value_counts())

#Check and save cleaned dataset
df.to_csv("Data/cleaned_pollution_dataset.csv", index=False)
