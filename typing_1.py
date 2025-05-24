import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
df = pd.read_csv("typing_data.csv")
print("Data loaded successfully!\n")

# Step 2: Check and Handle Missing Values
print("Missing values before handling:")
print(df.isnull().sum())
df = df.dropna()  # Drop rows with missing values
print("\nMissing values after dropping:")
print(df.isnull().sum())

# Step 3: Data Types and Conversion
df['Date'] = pd.to_datetime(df['Date'])
print("\nData types after conversion:")
print(df.dtypes)

# Step 4: Data Integrity - Remove rows with invalid WPM or Accuracy
df = df[(df['WPM'] > 0) & (df['Accuracy (%)'] >= 0) & (df['Accuracy (%)'] <= 100)]

# Step 5: Feature Engineering - Speed Category
df['Speed Category'] = pd.cut(df['WPM'], bins=[0, 40, 60, 1000], labels=['Slow', 'Average', 'Fast'])

# Step 6: Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Step 7: Outlier Detection with Boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=df['WPM'])
plt.title("Boxplot of Words Per Minute (WPM)")
plt.xlabel("WPM")
plt.savefig("boxplot_wpm.png")
plt.show()

# Step 8: Distribution of WPM - Histogram + KDE
plt.figure(figsize=(8,5))
sns.histplot(df['WPM'], bins=15, kde=True, color='skyblue')
plt.title("Distribution of WPM")
plt.xlabel("WPM")
plt.ylabel("Frequency")
plt.savefig("histogram_wpm.png")
plt.show()

# Step 9: Scatterplot WPM vs Accuracy with Speed Category Hue
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='WPM', y='Accuracy (%)', hue='Speed Category', palette='Set2')
plt.title("Typing Speed vs Accuracy")
plt.xlabel("WPM")
plt.ylabel("Accuracy (%)")
plt.savefig("scatter_wpm_accuracy.png")
plt.show()

# Step 10: Countplot of Speed Categories
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Speed Category', palette='Set1')
plt.title("Count of Speed Categories")
plt.xlabel("Speed Category")
plt.ylabel("Number of Samples")
plt.savefig("countplot_speed_category.png")
plt.show()

# Step 11: Trend of WPM Over Time
plt.figure(figsize=(10,4))
sns.lineplot(data=df.sort_values('Date'), x='Date', y='WPM', marker='o')
plt.title("Typing Speed (WPM) Over Time")
plt.xlabel("Date")
plt.ylabel("WPM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("lineplot_wpm_time.png")
plt.show()

print("\nEDA & Preprocessing completed. All plots saved as PNG files.")
