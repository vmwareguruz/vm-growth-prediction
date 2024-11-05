import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['vm_growth'] = data['number_of_vms'].pct_change()
    return data.dropna()

# Preprocess data and save
processed_data = preprocess_data('data/raw_data.csv')
processed_data.to_csv('data/processed_data.csv')
