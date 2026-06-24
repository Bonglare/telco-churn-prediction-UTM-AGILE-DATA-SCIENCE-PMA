import pandas as pd
import os

def load_data():
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(csv_path)
    return df

def test_data_shape():
    df = load_data()
    assert df.shape[0] > 0, "Dataset is empty"
    assert df.shape[1] == 21, "Dataset does not have 21 columns"

def test_no_missing_target():
    df = load_data()
    assert df['Churn'].isnull().sum() == 0, "Target variable has missing values"
