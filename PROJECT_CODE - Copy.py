import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(file_path)
print("Analyze crime data to find trends, hotspots, and types of crimes using EDA and visualization.")
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

"""CLEANING"""

df.dropna(inplace=True)
if "DATE OCC" in df.columns:
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], format="%m/%d/%Y", errors="coerce")
if "TIME OCC" in df.columns:
    df["TIME OCC"] = df["TIME OCC"].astype(str).str.zfill(4)
    df["Hour"] = df["TIME OCC"].str[:2].astype(int)
if "Vict Age" in df.columns:
    covariance = df[['Hour', 'Vict Age']].cov().iloc[0, 1]
    print(f"Covariance between Hour and Victim Age: {covariance}")
 
""""scatterploat"""
"""
if "Vict Age" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Hour', y='Vict Age', data=df, alpha=0.9,color='red')
    plt.title('Scatter Plot: Victim Age vs Crime Hour')
    plt.xlabel('Hour')
    plt.ylabel('Victim Age')
    plt.tight_layout()
    plt.show()
    """
if "Vict Age" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Hour', y='Vict Age', data=df, alpha=0.9, color='red', label='Individual Crimes')

    hourly_avg_age = df.groupby('Hour')['Vict Age'].mean().reset_index()
    sns.lineplot(x='Hour', y='Vict Age', data=hourly_avg_age, color='blue', linewidth=2, label='Avg Victim Age')

    plt.title('Scatter Plot with Line: Victim Age vs Crime Hour')
    plt.xlabel('Hour')
    plt.ylabel('Victim Age')
    plt.legend()
    plt.tight_layout()
    plt.show()
   
"""1. Line Chart - Monthly crime trend"""

df["YearMonth"] = df["DATE OCC"].dt.to_period('M')
monthly_crime = df["YearMonth"].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(monthly_crime.index.astype(str), monthly_crime.values, marker="o", color="darkblue")
plt.xticks(rotation=45)
plt.title("Monthly Crime Trend")
plt.xlabel("Month")
plt.ylabel("Crime Count")
plt.tight_layout()
plt.show()

"""2. Bar Chart - Top 10 Crime Types"""
plt.figure(figsize=(12, 6))
top_crimes = df["Crm Cd Desc"].value_counts().nlargest(10)
top_crimes.plot(kind="bar", color="orange")
plt.title("Top 10 Most Frequent Crime Types")
plt.xlabel("Crime Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""3. Histogram - Crimes by Hour of Day
plt.figure(figsize=(10, 5))
plt.hist(df["Hour"], bins=24, color="teal", edgecolor="black")
plt.title("Crime Occurrences by Hour")
plt.xlabel('Hour')
plt.ylabel('Crime Count')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()"""
"""3. Histogram with Line Chart - Crimes by Hour of Day"""
plt.figure(figsize=(10, 5))

hist_values, bins, _ = plt.hist(df["Hour"], bins=24, color="teal", edgecolor="black", alpha=0.6, label="Histogram")

bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.plot(bin_centers, hist_values, color="blue", marker="o", linewidth=2, label="Trend Line")

plt.title("Crime Occurrences by Hour with Trend Line")
plt.xlabel('Hour')
plt.ylabel('Crime Count')
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.show()
"""4. Box Plot - Crime count per area"""
plt.figure(figsize=(14, 6))
sns.boxplot(x='AREA NAME', y='Hour', data=df)
plt.xticks(rotation=90)
plt.title('Crime Hour Distribution by Area')
plt.tight_layout()
plt.show()


"""5. Heatmap - Hour vs Crime Count"""
heatmap_data = df.groupby(['Hour', 'AREA NAME']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap: Hour vs Area Crime Count')
plt.xlabel('Area')
plt.ylabel('Hour')
plt.tight_layout()
plt.show()

"""corelation HEATMAP"""

plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

"""6. Pie Chart - Crime distribution by area"""
area_crime = df['AREA NAME'].value_counts().nlargest(6)
plt.figure(figsize=(8, 8))
plt.pie(area_crime, labels=area_crime.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Top 6 Areas with Highest Crimes')
plt.tight_layout()
plt.show()

"""KDE Plot: Hour vs Crime Density"""
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df['Hour'], fill=True, color='blue')
plt.title('KDE Plot: Crime Density by Hour')
plt.xlabel('Hour')
plt.tight_layout()
plt.show()

"""Pairplot (on small sample due to memory)"""
sample = df[['Hour']].sample(1000, random_state=1, replace=True)

sns.pairplot(sample)
plt.suptitle('Pairplot Sample: Crime Hour', y=1.02)
plt.show()

