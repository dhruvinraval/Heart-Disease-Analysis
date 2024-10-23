```python
import pandas as pd

```

# Load the file


```python
# Load the CSV file
df = pd.read_csv("heart.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>241</td>
      <td>0</td>
      <td>1</td>
      <td>123</td>
      <td>1</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>45</td>
      <td>1</td>
      <td>3</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>68</td>
      <td>1</td>
      <td>0</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>1</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>57</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 14 columns</p>
</div>




```python
# Display the original column names and data types
print("Original Columns and Data Types:")
print(df.dtypes)

```

    Original Columns and Data Types:
    age           int64
    sex           int64
    cp            int64
    trtbps        int64
    chol          int64
    fbs           int64
    restecg       int64
    thalachh      int64
    exng          int64
    oldpeak     float64
    slp           int64
    caa           int64
    thall         int64
    output        int64
    dtype: object
    


```python
# Define a mapping for renaming the columns
column_renames = {
    'cp': 'chest_pain_types',  # Replace with actual old and new column names
    'trtbps': 'resting_blood_pressure',
    'chol': 'cholesterol',
    'fbs' : 'fasting_blood_sugar',
    'restecg' : 'resting_electrocardiographic_results',
    'thalachh' : 'max_heart_rate',
    'exng' : 'exercise_induced_angina',
    'slp' : 'slope',
    'thall' : 'thalassemia',
    'output' : 'heart_disease'
    # Add more columns as needed
}
```


```python
column_renames
```




    {'cp': 'chest_pain_types',
     'trtbps': 'resting_blood_pressure',
     'chol': 'cholesterol',
     'fbs': 'fasting_blood_sugar',
     'restecg': 'resting_electrocardiographic_results',
     'thalachh': 'max_heart_rate',
     'exng': 'exercise_induced_angina',
     'slp': 'slope',
     'thall': 'thalassemia',
     'output': 'heart_disease'}




```python
df.rename(columns=column_renames, inplace=True)
```


```python
print("\nUpdated Columns")
print(df.dtypes)
```

    
    Updated Columns
    age                                       int64
    sex                                       int64
    chest_pain_types                          int64
    resting_blood_pressure                    int64
    cholesterol                               int64
    fasting_blood_sugar                       int64
    resting_electrocardiographic_results      int64
    max_heart_rate                            int64
    exercise_induced_angina                   int64
    oldpeak                                 float64
    slope                                     int64
    caa                                       int64
    thalassemia                               int64
    heart_disease                             int64
    dtype: object
    


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>chest_pain_types</th>
      <th>resting_blood_pressure</th>
      <th>cholesterol</th>
      <th>fasting_blood_sugar</th>
      <th>resting_electrocardiographic_results</th>
      <th>max_heart_rate</th>
      <th>exercise_induced_angina</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>caa</th>
      <th>thalassemia</th>
      <th>heart_disease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>241</td>
      <td>0</td>
      <td>1</td>
      <td>123</td>
      <td>1</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>45</td>
      <td>1</td>
      <td>3</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>68</td>
      <td>1</td>
      <td>0</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>1</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>57</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 14 columns</p>
</div>




```python
# Replace integer values in the 'sex','fasting_blood_sugar  (1 = true, 0 = false)','Exercise induced angina',
#'Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)','heart disease (1='yes',0='no')' column with text labels

df['sex'] = df['sex'].replace({0: 'Female', 1: 'Male'})
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].replace({0:'False',1:'True'})
df['thalassemia'] = df ['thalassemia'].replace({1:'Normal',2:'Fixed defect',3:'Reversible defect'})
df['heart_disease'] = df['heart_disease'].replace({1:'Yes', 0:'No'})
print("\nUpdated Values for sex with First 5 Rows:")
print(df.head())
```

    
    Updated Values for sex with First 5 Rows:
       age     sex  chest_pain_types  resting_blood_pressure  cholesterol  \
    0   63    Male                 3                     145          233   
    1   37    Male                 2                     130          250   
    2   41  Female                 1                     130          204   
    3   56    Male                 1                     120          236   
    4   57  Female                 0                     120          354   
    
      fasting_blood_sugar  resting_electrocardiographic_results  max_heart_rate  \
    0                True                                     0             150   
    1               False                                     1             187   
    2               False                                     0             172   
    3               False                                     1             178   
    4               False                                     1             163   
    
       exercise_induced_angina  oldpeak  slope  caa   thalassemia heart_disease  
    0                        0      2.3      0    0        Normal           Yes  
    1                        0      3.5      0    0  Fixed defect           Yes  
    2                        0      1.4      2    0  Fixed defect           Yes  
    3                        0      0.8      2    0  Fixed defect           Yes  
    4                        1      0.6      2    0  Fixed defect           Yes  
    


```python
print(df.dtypes)
```

    age                                       int64
    sex                                      object
    chest_pain_types                          int64
    resting_blood_pressure                    int64
    cholesterol                               int64
    fasting_blood_sugar                      object
    resting_electrocardiographic_results      int64
    max_heart_rate                            int64
    exercise_induced_angina                   int64
    oldpeak                                 float64
    slope                                     int64
    caa                                       int64
    thalassemia                              object
    heart_disease                            object
    dtype: object
    


```python
# Save the updated DataFrame to a new CSV file
df.to_csv('heart_disease.csv', index=False)
```


```python

```

# Exploratory Data Analysis (EDA)


```python
# Get summary statistics for numerical columns
print(df.describe())
```

                  age  chest_pain_types  resting_blood_pressure  cholesterol  \
    count  303.000000        303.000000              303.000000   303.000000   
    mean    54.366337          0.966997              131.623762   246.264026   
    std      9.082101          1.032052               17.538143    51.830751   
    min     29.000000          0.000000               94.000000   126.000000   
    25%     47.500000          0.000000              120.000000   211.000000   
    50%     55.000000          1.000000              130.000000   240.000000   
    75%     61.000000          2.000000              140.000000   274.500000   
    max     77.000000          3.000000              200.000000   564.000000   
    
           resting_electrocardiographic_results  max_heart_rate  \
    count                            303.000000      303.000000   
    mean                               0.528053      149.646865   
    std                                0.525860       22.905161   
    min                                0.000000       71.000000   
    25%                                0.000000      133.500000   
    50%                                1.000000      153.000000   
    75%                                1.000000      166.000000   
    max                                2.000000      202.000000   
    
           exercise_induced_angina     oldpeak       slope         caa  
    count               303.000000  303.000000  303.000000  303.000000  
    mean                  0.326733    1.039604    1.399340    0.729373  
    std                   0.469794    1.161075    0.616226    1.022606  
    min                   0.000000    0.000000    0.000000    0.000000  
    25%                   0.000000    0.000000    1.000000    0.000000  
    50%                   0.000000    0.800000    1.000000    0.000000  
    75%                   1.000000    1.600000    2.000000    1.000000  
    max                   1.000000    6.200000    2.000000    4.000000  
    


```python
# Check for missing values in the dataset
print(df.isnull().sum())
```

    age                                     0
    sex                                     0
    chest_pain_types                        0
    resting_blood_pressure                  0
    cholesterol                             0
    fasting_blood_sugar                     0
    resting_electrocardiographic_results    0
    max_heart_rate                          0
    exercise_induced_angina                 0
    oldpeak                                 0
    slope                                   0
    caa                                     0
    thalassemia                             0
    heart_disease                           0
    dtype: int64
    


```python
import matplotlib.pyplot as plt
```


```python
#Data distribution

# Histogram PLot for age
df['age'].hist(bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

```


    
![png](output_17_0.png)
    



```python
# Check the distribution of the 'sex' column
print(df['sex'].value_counts())
```

    Male      207
    Female     96
    Name: sex, dtype: int64
    

## Visualizing Data with matplotlib and seaborn


```python
import seaborn as sns
```

##### Visualize the correlation between different numerical variables to identify relationships between them.


```python
# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```


    
![png](output_22_0.png)
    


##### The code calculates the correlation matrix for the variables of the heart_disease.csv dataset, showing the relationships between different heart-related features using pairwise correlation coefficients.
##### It visualizes the correlation matrix as a heatmap with annotated values and color gradients, helping to identify positive and negative correlations between the heart-related variables in the dataset. 

### Box Plot for Outlier Detection
##### A box plot helps identify outliers and understand the spread of data for numeric columns.


```python
# Boxplot for cholesterol levels
sns.boxplot(x=df['cholesterol'])
plt.title('Cholesterol Levels')
plt.show()
```


    
![png](output_25_0.png)
    


###  Scatter Plot to Visualize Relationships


```python
# Scatter plot for age vs cholesterol
plt.scatter(df['age'], df['cholesterol'])
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age vs Cholesterol')
plt.show()
```


    
![png](output_27_0.png)
    


###  Grouping and Aggregation
##### Group the data by categorical columns and apply aggregate functions to analyze the data. 


```python
# Group by sex and calculate mean age and cholesterol
grouped = df.groupby('sex').agg({'age': 'mean', 'cholesterol': 'mean'})
print(grouped)

```

                  age  cholesterol
    sex                           
    Female  55.677083   261.302083
    Male    53.758454   239.289855
    

### Feature Engineering
##### Categorizing age into bins or age groups.


```python
# Categorize age into age groups
bins = [0, 30, 50, 70, 100]
labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
print(df[['age', 'age_group']].head())
```

       age    age_group
    0   63       Senior
    1   37  Middle-aged
    2   41  Middle-aged
    3   56       Senior
    4   57       Senior
    


```python
print(df[['age_group','heart_disease']].head())
```

         age_group heart_disease
    0       Senior           Yes
    1  Middle-aged           Yes
    2  Middle-aged           Yes
    3       Senior           Yes
    4       Senior           Yes
    

##### Comparing "Age Group" and "Heart Disease"


```python
# First, count the occurrences of heart disease in each age group
# 'heart_disease' is 1 for yes and 0 for no
heart_disease_count = df.groupby(['age_group', 'heart_disease']).size().reset_index(name='count')
heart_disease_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_group</th>
      <th>heart_disease</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Young</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Young</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Middle-aged</td>
      <td>No</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Middle-aged</td>
      <td>Yes</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Senior</td>
      <td>No</td>
      <td>108</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Senior</td>
      <td>Yes</td>
      <td>94</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Elderly</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Elderly</td>
      <td>Yes</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
# Plotting with Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='age_group', y='count', hue='heart_disease', data=heart_disease_count, palette='Set1')

# Add title and labels
plt.title('Comparison of Heart Disease Across Age Groups', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])

# Display the plot
plt.show()
```


    
![png](output_36_0.png)
    


### Building a Simple Model

#### Logistic Regression


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
# Load your dataset
df2 = pd.read_csv('heart.csv')

# Define your features (X) and target (y)
X = df2.drop('output', axis=1)  # Features (drop the target column)
y = df2['output']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

    Model Accuracy: 0.81
    

#### The model accuracy of 0.81 means that the logistic regression model correctly predicted whether or not someone has heart disease for 81% of the cases in the dataset. 

### Confusion matrix 


```python
from sklearn.metrics import confusion_matrix
```


```python
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

    [[32  9]
     [ 8 42]]
    


```python
from sklearn.metrics import classification_report

# Get precision, recall, and F1-score
print(classification_report(y_test, y_pred))

```

                  precision    recall  f1-score   support
    
               0       0.80      0.78      0.79        41
               1       0.82      0.84      0.83        50
    
        accuracy                           0.81        91
       macro avg       0.81      0.81      0.81        91
    weighted avg       0.81      0.81      0.81        91
    
    


```python

```
