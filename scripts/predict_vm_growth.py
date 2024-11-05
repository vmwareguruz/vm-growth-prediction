import pandas as pd
import joblib

def predict_vm_growth(input_data):
    model = joblib.load('models/vm_growth_model.pkl')
    predictions = model.predict(input_data)
    return predictions

# Example usage
input_data = pd.DataFrame({
    'month': [4],
    'year': [2023],
    'vm_growth': [0.15]  # Example growth rate
})
print(predict_vm_growth(input_data))
