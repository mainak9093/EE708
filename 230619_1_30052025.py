#Author : Mainak Sarkar 230619

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:/EE 708/A1.csv")#Reading the doc

print("Original columns:", df.columns.tolist())
df.columns = df.columns.str.strip()
print("Cleaned columns: ", df.columns.tolist())

df = df.rename(columns={
    'Classes':  'Class', 
    'Feature 1':'Feature1',
    'Feature 2':'Feature2',
    'Feature 3':'Feature3',
    'Feature 4':'Feature4'
})

print("Renamed columns:", df.columns.tolist())
#6.a Finding Frequency of each class sample
print("\nFrequency of samples for each class:")
print(df['Class'].value_counts())
#6.b Data description and  interquartile range for all four features
desc = df[['Feature1','Feature2','Feature3','Feature4']].describe()
print("\nData Description:\n", desc)
iqr = desc.loc['75%'] - desc.loc['25%']
print("\nInterquartile Range (IQR):\n", iqr)
#6.c  Ploting histogram of feature 1 for class A. 
plt.figure(figsize=(6,4))
df[df['Class']=='A']['Feature1'].hist(bins=20)
plt.title('Histogram of Feature1 for Class A')
plt.xlabel('Feature1'); plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#6.d Ploting  box plot for feature 2 for each class
plt.figure(figsize=(6,4))
sns.boxplot(x='Class', y='Feature2', data=df)
plt.title('Box Plot: Feature2 by Class')
plt.show()
# 6.e Plotting Violin plot for feature 3 for each class
plt.figure(figsize=(6,4))
sns.violinplot(x='Class', y='Feature3', data=df)
plt.title('Violin Plot: Feature3 by Class')
plt.show()
# 6.f Plotting Scatter plots between feature 1 and feature 3
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='Feature1', y='Feature3', hue='Class')
plt.title('Scatter Plot: Feature1 vs Feature3 by Class')
plt.show()
# 6.g Plotting Contour plot between feature 1 and feature 4
plt.figure(figsize=(6,4))
for c in df['Class'].unique():
    sub = df[df['Class']==c]
    sns.kdeplot(x=sub['Feature1'], y=sub['Feature4'],
                fill=True, alpha=0.4, label=f"Class {c}")
plt.title('Contour Plot: Feature1 vs Feature4 by Class')
plt.legend()
plt.show()
# 6.g Plotting Hexagonal bin plot for class A between features 2 and 4
plt.figure(figsize=(6,4))
subA = df[df['Class']=='A']
plt.hexbin(subA['Feature2'], subA['Feature4'], gridsize=25)
plt.colorbar(label='Count')
plt.xlabel('Feature2'); plt.ylabel('Feature4')
plt.title('Hexbin: Feature2 vs Feature4 (Class A)')
plt.show()
# 6.g Correlation matrix for the four features.
corr = df[['Feature1','Feature2','Feature3','Feature4']].corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# 6.h Pair plot for the four features
sns.pairplot(df, vars=['Feature1','Feature2','Feature3','Feature4'],
             hue='Class', diag_kind='kde')
plt.suptitle('Pair Plot of Features by Class', y=1.02)
plt.show()