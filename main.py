import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load data
df = pd.read_csv('train.csv')
print(df.info())
print(df.head())

# 2. Pie chart to display survival rates
plt.figure(figsize=(8, 6))
plt.pie(df['Survived'].value_counts(), labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['purple', 'orange'])
plt.title('Survival Rate')
plt.axis('equal')
plt.show()
# This chart shows the proportion of survivors in the dataset with updated colors for visual distinction.

# 3. Bar chart of survival rates by sex
df['Survived'] = df['Survived'].astype(int)
survived_by_sex = df.groupby('Sex', as_index=False)['Survived'].mean()
plt.figure(figsize=(8, 6))
plt.bar(survived_by_sex['Sex'], survived_by_sex['Survived'], color=['lightgreen', 'lightblue'])
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Sex')
plt.show()
# Majority of women survived (over 70%). Among men, the survival rate was much lower, around 20%.

# 4. Bar chart for missing values by feature
missing_values = df.isnull().sum()
missing_values_sorted = missing_values.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
missing_values_sorted.plot(kind='bar', color='violet')
plt.title('Missing Values by Feature (sorted)')
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 'Cabin' has the most missing values, nearly 700, shown in violet. 'Age' also has significant missing values, about 175. 'Embarked' has fewer than 10 missing values, which is minimal.

# 5. Violin plot to explore the relationship between class, age, and survival
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=df, split=True, palette='cool')
plt.title('Pclass and Age vs. Survived')
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])
plt.yticks(range(0, 100, 10))
plt.show()
# First class predominantly shows middle-aged survivors. In second and third classes, a larger fraction of younger individuals survived.

# 6. Histogram to examine age distribution
age_values = df['Age'].dropna()
plt.figure(figsize=(10, 6))
plt.hist(age_values, bins=30, color='coral', edgecolor='black', alpha=0.7)
plt.axvline(age_values.mean(), color='blue', linewidth=1, label='Mean Age')
plt.axvline(np.median(age_values), color='green', linewidth=1, label='Median Age')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# The majority of passengers are aged 20-40. The median and mean ages are approximately 30 years, indicated by green and blue lines respectively.

# 7. Violin plot to examine how embarkation point and age relate to survival
# Яким чином вік і місце посадки пасажирів корелювали з шансом вижити.
sns.violinplot(x="Embarked", y="Age", hue="Survived", data=df, split=True, palette='pastel')
plt.title('Embarkation Point and Age vs. Survived')
plt.xticks([0, 1, 2], ['Southampton', 'Cherbourg', 'Queenstown'])
plt.yticks(range(0, 100, 10))
plt.show()
# Most children embarking from Southampton (S) and Cherbourg (C) survived. In Queenstown (Q), a greater proportion of middle-aged individuals survived.

