#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Checking for missing values before forecasting


# In[21]:


import pandas as pd

# Load data and set index
df = pd.read_csv('Cleaned Data for python forecasting.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.set_index('Date')

# Check for missing values
missing_values = df.isna().sum()
print(missing_values)


# In[ ]:


#Here we change the missing value into the mean of the respective column


# In[10]:


#After the change the data is saved 


# In[16]:


import pandas as pd

# read in your data
df = pd.read_csv('Cleaned Data for python forecasting.csv')

# Select only numeric columns and replace missing values with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# In[17]:


df.to_csv('New Data weather.csv', index=False)


# In[ ]:


#After the change we save the data in another name inorder to apply the new changes
#Then we recheck to see if the error persists


# In[18]:


import pandas as pd

# Load data and set index
df = pd.read_csv('New Data weather.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.set_index('Date')

# Check for missing values
missing_values = df.isna().sum()
print(missing_values)


# In[ ]:


# The main forecasting happens from this point.
# All the codes have been given the neccessary explanation


# In[19]:


import pandas as pd
from pmdarima.arima import auto_arima

# Load data and set index
df = pd.read_csv('New Data weather.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.set_index('Date')

# Define seasonal order for SARIMA model
seasonal_order = (1, 1, 1, 12)

# Loop through each column and fit SARIMA model, forecast, and concatenate with original data
columns = ['Temperature', 'Average humidity (%)', 'Average dewpoint (°F)', 'Average barometer (in)',
           'Average windspeed (mph)', 'Average gustspeed (mph)', 'Average direction (°deg)',
           'Rainfall for month (in)', 'Rainfall for year (in)', 'Maximum rain per minute',
           'Maximum temperature (°F)', 'Minimum temperature (°F)', 'Maximum humidity (%)',
           'Minimum humidity (%)', 'Maximum pressure', 'Minimum pressure', 'Maximum windspeed (mph)',
           'Maximum gust speed (mph)', 'Maximum heat index (°F)', 'Month', 'diff_pressure']
forecast_dfs = []
for col in columns:
    # Fit SARIMA model and forecast
    model = auto_arima(df[col], seasonal=True, m=12, suppress_warnings=True)
    forecast = model.predict(n_periods=4748)
    forecast_df = pd.DataFrame(forecast, columns=[col])
    forecast_df.index = pd.date_range(start='01-01-2023', periods=4748, freq='D')
    forecast_dfs.append(forecast_df)

# Concatenate original and forecast dataframes
combined_df = pd.concat([df] + forecast_dfs, axis=1)
print(combined_df.head())


# In[ ]:


# Here we combine the original and the forecasted dataset and downloaded in csv format


# In[20]:


combined_df.to_csv('Completed weather forecasting data.csv', index=False)


# In[ ]:




