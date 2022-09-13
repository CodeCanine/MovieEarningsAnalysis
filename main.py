import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Setting up display for pandas visualisation
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

# Reading in dataset
df = pd.read_csv('movies.csv')

# Checking dataset layout
print(df.head())

# Checking if there are null values in any columns
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))

# Checking datatypes of columns
print(df.dtypes)

# Changing NaN values to 0
df['budget'] = df['budget'].fillna(0)
df['gross'] = df['gross'].fillna(0)

# With no NaN values, converting the Gross earnings and Budget to long int
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

# Double-checking the converted columns
print(df.dtypes)

# The year column is not always correct, using regex to extract the year, fill the NaN values, and converting to int
df['year_correct'] = df['released'].str.extract(r'(\d{4})')
df['year_correct'] = df['year_correct'].fillna(0)
df['year_correct'] = df['year_correct'].astype(int)

# Sorting the values by gross earnings
df = df.sort_values(by=['gross'], inplace=False, ascending=False)
print(df)

# Visualizing the correlation between gross earnings and budget
plt.figure(figsize=(10, 8), dpi=150)
sns.scatterplot(data=df, x='budget', y='gross')
plt.title('Budget vs. Gross earnings')
plt.xlabel('Gross Earnings (Hundred Millions)')
plt.ylabel('Budget for Film (Billions)')
plt.savefig('scatter', bbox_inches='tight')
plt.show()

# Further refining the visualisation correlation between gross earnings and budget
plt.figure(figsize=(10, 8), dpi=150)
sns.regplot(data=df, x='budget', y='gross', scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.xlabel('Gross Earnings (Hundred Millions)')
plt.ylabel('Budget for Film (Billions)')
plt.savefig('reg.png', bbox_inches='tight')
plt.show()

# Visualizing correlation on heatmap, between all int values
correlation_matrix = df.corr()
plt.figure(figsize=(15, 12), dpi=150)
sns.heatmap(data=correlation_matrix, annot=True, annot_kws={"fontsize": 12.5})
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.savefig('heat_corr.png', bbox_inches='tight')
plt.show()

# Creating copy before further calculations
df_numerized = df.copy()

# Converting object type data to category, to numerize the non int values for calculations
for col_name in df_numerized.columns:
    if df_numerized[col_name].dtypes == 'object':
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

# Double-checking result
print(df_numerized)

# Creating correlation between all categories, to see what is the highest
plt.figure(figsize=(15, 12), dpi=150)
correlation_matrix = df_numerized.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, annot_kws={"fontsize": 12.5})
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.savefig('heat_corr_all.png', bbox_inches='tight')
plt.show()

# Creating correlation pairs
correlation_matrix_numerized = df_numerized.corr()
corr_pairs = correlation_matrix.unstack()
print(corr_pairs)

# Sorting pairs in ascending order
sorted_pairs = corr_pairs.sort_values()
print(sorted_pairs)

# Checking the highest correlations, excluding same column correlations
high_corr = (sorted_pairs[(sorted_pairs > 0.5) & (sorted_pairs < 1)])
print(high_corr)

# Conclusion:
# The Budget has the highest relationship to the Gross earnings
