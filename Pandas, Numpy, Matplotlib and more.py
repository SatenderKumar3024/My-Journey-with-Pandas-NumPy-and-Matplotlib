#!/usr/bin/env python
# coding: utf-8

# # Pandas, Numpy, Matplotlib, seaborn and more

# In[39]:


# Importing necessary libraries for data analysis
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For statistical data visualization based on Matplotlib
import scipy  # For scientific and technical computing (including optimization, integration, and statistics)
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preprocessing data (scaling, encoding)
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # For linear regression models
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation metrics
import statsmodels.api as sm  # For statistical modeling and hypothesis testing


# In[40]:


dict1 = {
    "name":['satender', 'rohan', 'rahul', 'ajay'],
    "marks":[92, 34, 34, 17],
    "city":['london', 'ontario', 'ajex', 'windsor']
}


# In[41]:


df = pd.DataFrame(dict1)


# In[42]:


df


# In[43]:


import pandas as pd
import numpy as np

dict1 = {
    "name": ['satender', 'rohan', 'rahul', 'ajay'],
    "marks": [92, 34, 34, 17],
    "city": ['london', 'ontario', 'ajex', 'windsor']
}


# In[44]:


df = pd.DataFrame(dict1)


# In[45]:


df


# In[46]:


import pandas as pd
import numpy as np

# Define the DataFrame
dict1 = {
    "name": ['satender', 'rohan', 'rahul', 'ajay'],
    "marks": [92, 34, 34, 17],
    "city": ['london', 'ontario', 'ajex', 'windsor']
}

df = pd.DataFrame(dict1)

# Add a calculated column 'grade' based on marks
df['grade'] = np.select(
    [
        df['marks'] >= 90, 
        df['marks'] >= 75, 
        df['marks'] >= 50, 
        df['marks'] >= 35
    ],
    ['A', 'B', 'C', 'D'],
    default='F'
)


# In[47]:


print(df)


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of students in each grade
grade_counts = df['grade'].value_counts()

# Create a bar plot with the updated syntax
plt.figure(figsize=(8, 6))
sns.barplot(x=grade_counts.index, y=grade_counts.values, hue=grade_counts.index, dodge=False, palette='viridis', legend=False)
plt.title('Distribution of Grades')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.show()


# In[49]:


import pandas as pd
import numpy as np

# Original DataFrame
dict1 = {
    "name": ['satender', 'rohan', 'rahul', 'ajay'],
    "marks": [92, 34, 34, 17],
    "city": ['london', 'ontario', 'ajex', 'windsor']
}
df = pd.DataFrame(dict1)

# Add a column for pass/fail status based on marks
df['status'] = df['marks'].apply(lambda x: 'Pass' if x >= 35 else 'Fail')

# Add a column for a custom grade based on multiple criteria
def custom_grade(row):
    if row['marks'] >= 90:
        return 'A+'
    elif row['marks'] >= 75:
        return 'A'
    elif row['marks'] >= 50:
        return 'B'
    elif row['marks'] >= 35:
        return 'C'
    else:
        return 'F'

df['custom_grade'] = df.apply(custom_grade, axis=1)

print(df)


# In[50]:


# Group by city and calculate the mean, sum, and count of marks
city_group = df.groupby('city')['marks'].agg(['mean', 'sum', 'count'])

# Add a column for the rank of each student within their city
df['rank_within_city'] = df.groupby('city')['marks'].rank(ascending=False)

print(city_group)
print(df)


# In[51]:


# Create a pivot table to show the average marks and count of students by city and status
pivot_table = pd.pivot_table(df, values='marks', index='city', columns='status', aggfunc={'marks': ['mean', 'count']})

print(pivot_table)


# In[52]:


# Additional DataFrame with more student information
additional_info = pd.DataFrame({
    'name': ['satender', 'rohan', 'rahul', 'ajay', 'vikram'],
    'age': [22, 21, 22, 23, 24],
    'gender': ['M', 'M', 'M', 'M', 'M']
})

# Merge on the 'name' column
merged_df = pd.merge(df, additional_info, on='name', how='left')

print(merged_df)


# In[53]:


# Introduce some missing data for demonstration
merged_df.loc[merged_df['name'] == 'vikram', 'marks'] = np.nan

# Fill missing data with the mean of the column
merged_df['marks_filled'] = merged_df['marks'].fillna(merged_df['marks'].mean())

# Alternatively, interpolate missing values based on surrounding data
merged_df['marks_interpolated'] = merged_df['marks'].interpolate()

print(merged_df)


# In[54]:


# Filter students who passed and scored more than the average marks of the class
average_marks = df['marks'].mean()
filtered_students = df[(df['status'] == 'Pass') & (df['marks'] > average_marks)]

print(filtered_students)


# In[55]:


# Calculate a rolling average of marks with a window size of 2
df['rolling_avg'] = df['marks'].rolling(window=2).mean()

print(df)


# In[56]:


# Custom function to generate a summary for each row
def summarize_row(row):
    return f"{row['name']} from {row['city']} scored {row['marks']} and got grade {row['custom_grade']}."

df['summary'] = df.apply(summarize_row, axis=1)

print(df['summary'])


# In[57]:


import matplotlib.pyplot as plt

# Bar plot for average marks by city
df.groupby('city')['marks'].mean().plot(kind='bar', title='Average Marks by City')
plt.ylabel('Average Marks')
plt.show()

# Pie chart for distribution of grades
df['custom_grade'].value_counts().plot(kind='pie', title='Grade Distribution', autopct='%1.1f%%')
plt.ylabel('')
plt.show()


# In[58]:


# Set a MultiIndex with 'city' and 'name'
df_multiindex = df.set_index(['city', 'name'])

# Access data using the MultiIndex
print(df_multiindex.loc['london'].loc['satender'])

# Reset the index to default
df_reset = df_multiindex.reset_index()
print(df_reset)


# In[59]:


# Convert 'custom_grade' to a categorical type with order
grades = ['F', 'D', 'C', 'B', 'A', 'A+']
df['custom_grade'] = pd.Categorical(df['custom_grade'], categories=grades, ordered=True)

# Sort DataFrame by categorical grade
df_sorted_by_grade = df.sort_values('custom_grade')

print(df_sorted_by_grade)


# In[60]:


plt.figure(figsize=(8, 6))
sns.histplot(df['marks'], bins=10, kde=True, color='blue')
plt.title('Marks Distribution')
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.show()


# In[61]:


plt.figure(figsize=(8, 8))
plt.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Grade Distribution')
plt.show()


# In[66]:


# Binning an array of data
data = np.random.randn(1000)  # Generate 1000 random numbers
bins = np.linspace(-3, 3, 7)  # Create 6 bins between -3 and 3
digitized = np.digitize(data, bins)

# Counting the number of elements in each bin
bin_counts = np.bincount(digitized)
print("Bin Counts:", bin_counts)


# In[67]:


import numpy as np

# Vectorized operation using ufunc
arr = np.array([1, 2, 3, 4, 5])

# Example of vectorized operation
result = np.exp(arr)  # Exponential function applied element-wise
print("Exponential of Array:", result)

# Custom ufunc
def custom_func(x, y):
    return x + y * y

# Vectorized using np.vectorize
vectorized_func = np.vectorize(custom_func)
result = vectorized_func(arr, arr)
print("Custom Ufunc Result:", result)


# In[68]:


# Example of broadcasting without memory duplication
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([4, 5, 6])

# Broadcasting adds arr2 to each row of arr1 without copying data
result = arr1 + arr2
print("Broadcasted Result:\n", result)

# Memory-efficient operation
print("Memory Address of arr1:", arr1.__array_interface__['data'])
print("Memory Address of arr2:", arr2.__array_interface__['data'])


# In[69]:


# Integer array indexing
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("Original Array:\n", arr)

# Select elements using an integer array
result = arr[[0, 1, 2], [1, 0, 1]]  # Selects (0,1), (1,0), and (2,1)
print("Integer Array Indexing Result:", result)

# Boolean indexing
mask = arr > 2
print("Boolean Mask:\n", mask)
result = arr[mask]
print("Boolean Indexing Result:", result)


# In[70]:


# Create a structured array with multiple data types
data = np.array([
    ('Satender', 92, 'London'),
    ('Rohan', 34, 'Ontario'),
    ('Rahul', 34, 'Ajex'),
    ('Ajay', 17, 'Windsor')
], dtype=[('name', 'U10'), ('marks', 'i4'), ('city', 'U10')])

print("Structured Array:\n", data)

# Accessing specific fields
print("Names:", data['name'])
print("Marks:", data['marks'])


# In[71]:


# Create a masked array
arr = np.array([1, 2, 3, -1, 5])
masked_arr = np.ma.masked_array(arr, mask=[0, 0, 0, 1, 0])

print("Masked Array:", masked_arr)
print("Mean ignoring masked elements:", np.ma.mean(masked_arr))


# In[72]:


# Fancy indexing with arrays of indices
arr = np.arange(10)
print("Original Array:", arr)

# Fancy indexing to reorder elements
indices = [3, 1, 9, 7]
result = arr[indices]
print("Fancy Indexing Result:", result)

# Multi-indexing with slices
multi_indexed_result = arr[::2]  # Access every second element
print("Multi-Indexing Result:", multi_indexed_result)


# In[73]:


# Stacking arrays vertically and horizontally
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Vertical stacking
vstacked = np.vstack((arr1, arr2))
print("Vertically Stacked Arrays:\n", vstacked)

# Horizontal stacking
hstacked = np.hstack((arr1, arr2))
print("Horizontally Stacked Arrays:\n", hstacked)

# Splitting arrays
split_arr = np.hsplit(hstacked, 2)
print("Horizontally Split Arrays:", split_arr)


# In[74]:


from numpy.linalg import inv, eig, solve

# Define a matrix
matrix = np.array([[1, 2], [3, 4]])

# Inverse of a matrix
matrix_inv = inv(matrix)
print("Inverse of Matrix:\n", matrix_inv)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Solving a system of linear equations
# For example, solving Ax = b, where A is the matrix, and b is a vector
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = solve(A, b)
print("Solution of Linear Equations:", x)


# In[75]:


# Binning an array of data
data = np.random.randn(1000)  # Generate 1000 random numbers
bins = np.linspace(-3, 3, 7)  # Create 6 bins between -3 and 3
digitized = np.digitize(data, bins)

# Counting the number of elements in each bin
bin_counts = np.bincount(digitized)
print("Bin Counts:", bin_counts)


# In[77]:


# Generate a sample signal
time = np.linspace(0, 1, 400)
signal = np.sin(2 * np.pi * 50 * time) + np.sin(2 * np.pi * 120 * time)

# Perform FFT
signal_fft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=time[1] - time[0])

# Plot the FFT result
import matplotlib.pyplot as plt
plt.plot(frequencies, np.abs(signal_fft))
plt.title('FFT of the Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()


# In[78]:


from scipy import optimize

# Define a simple quadratic function
def f(x):
    return x**2 + 10*np.sin(x)

# Find the minimum of the function
result = optimize.minimize(f, x0=0)  # Start the search at x=0
print("Function minimum:", result.x)

# Numerical integration
from scipy.integrate import quad

result, error = quad(np.sin, 0, np.pi)
print("Integral of sin(x) from 0 to pi:", result)


# In[79]:


# Normalizing data
data = np.random.rand(100, 3)  # Random dataset with 3 features
normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
print("Normalized Data:\n", normalized_data)

# Generating polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(data)
print("Polynomial Features:\n", poly_features)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




