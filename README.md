# MovieEarningsAnalysis

Project to see what has the most impact on movies' gross earnings.
Dataset downloaded from www.kaggle.com
Source: https://www.kaggle.com/datasets/danielgrijalvas/movies?resource=download

<b>Approach:</b>

- Download dataset from kaggle
- Review and Modify data, to create accurate visualization
- Create multiple plots
- Analyze the plots
- Make a conclusion

<b>Plots:</b>

# 1 Visualizing the correlation between gross earnings and budget on Scatter Plot.
Plot clearly, however not visually appealingly reveal correlation between Budget and Gross income.
![scatter](https://user-images.githubusercontent.com/9075212/189919052-289eee47-2495-470c-883c-6d279711d6c2.png)

# 2 Further refining the visualisation between Gross earnings and Budget.
Refining scatter plot with regression plot, which shows a clear and exponential relationship between Budget and Gross income
![reg](https://user-images.githubusercontent.com/9075212/189919070-db073ac8-0460-465a-a9fa-a0e82ab8de16.png)

# 3 Visualizing correlation on heatmap, between all int values.
Heatmap shows the highest correlation between Budget and Gross earning in dataset, with current numeric values.
![heat_corr](https://user-images.githubusercontent.com/9075212/189919084-1dbb3a46-f4f6-4f57-a18e-db17a1225904.png)

# 4 Reviewing correlation between all categories, to see what is the highest
After converting the 'object' type columns to categories, digitalizing them, with the extended datasource for the heatmap,
the highest impact still seem to be the Budget to the Gross income.
![heat_corr_all](https://user-images.githubusercontent.com/9075212/189919090-c0c5782f-f25d-4204-8ee8-8b7886833c0e.png)

<b>Conclusion:</b>
As suspected, the highest impact on the Income seems to be the Budget for the movie, based on current dataset.
