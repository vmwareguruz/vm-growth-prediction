import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(data_path):
    data = pd.read_csv(data_path)
    X = data[['month', 'year', 'vm_growth']]
    y = data['number_of_vms']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    joblib.dump(model, 'models/vm_growth_model.pkl')

# Train the model
train_model('data/processed_data.csv')
