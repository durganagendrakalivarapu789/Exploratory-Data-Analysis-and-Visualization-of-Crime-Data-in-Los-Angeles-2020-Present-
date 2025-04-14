import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'Crime_Data_from_2020_to_Present.csv'
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Drop missing values
df.dropna(inplace=True)

# Convert DATE OCC to datetime
if 'DATE OCC' in df.columns:
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
    df.dropna(subset=['DATE OCC'], inplace=True)
    df['YearMonth'] = df['DATE OCC'].dt.to_period('M')
    df['Year'] = df['DATE OCC'].dt.year
    df['Month'] = df['DATE OCC'].dt.month

# Count of top 10 crimes - Bar Chart
plt.figure(figsize=(12, 6))
top_crimes = df['Crm Cd Desc'].value_counts().head(10)
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette='viridis')
plt.title('Top 10 Most Frequent Crimes')
plt.xlabel('Number of Crimes')
plt.ylabel('Crime Type')
plt.show()

# Crime trend over time - Line Chart
if 'YearMonth' in df.columns:
    monthly_crime = df['YearMonth'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    monthly_crime.plot(kind='line', marker='o', color='blue')
    plt.title('Monthly Crime Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Crime by Hour of Day - Histogram
if 'TIME OCC' in df.columns:
    df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
    df['Hour'] = df['TIME OCC'].str[:2].astype(int)
    plt.figure(figsize=(12, 6))
    plt.hist(df['Hour'], bins=24, color='orange', edgecolor='black')
    plt.title('Crime Occurrence by Hour of the Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Crimes')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.show()

# Crime by Area - Pie Chart
if 'AREA NAME' in df.columns:
    area_counts = df['AREA NAME'].value_counts().head(10)
    plt.figure(figsize=(8, 8))
    plt.pie(area_counts, labels=area_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Crime Distribution by Top 10 Areas')
    plt.axis('equal')
    plt.show()

# Crime count by year - Boxplot
if 'Year' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Year', y='Hour', data=df, palette='Set3')
    plt.title('Distribution of Crime Hours by Year')
    plt.xlabel('Year')
    plt.ylabel('Hour of Day')
    plt.show()

# Correlation Heatmap (only numeric columns)
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
