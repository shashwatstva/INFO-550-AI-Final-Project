import numpy as np
import pandas as pd

##Loading the raw dataset
data = pd.read_csv('seattle-weather.csv')

data.info()

# Creating a Subset with 500 Records
subset = data.sample(n=500, random_state=42)  # Randomly sample 500 records

# saving this subset to a CSV file:
subset.to_csv('subset.csv', index=False)
print('subset.csv created')

data.to_csv('alldata.csv', index= False)