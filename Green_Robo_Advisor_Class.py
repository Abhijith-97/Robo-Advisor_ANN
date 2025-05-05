# Green_Robo_Advisor_Class.py

import pandas as pd

class RoboAdvisor:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)

    def preprocess_data(self):
        """Basic preprocessing steps like handling missing values"""
        # Remove missing values (you can also apply other strategies)
        self.data = self.data.dropna()
        return self.data

    def feature_engineering(self):
        """Basic feature engineering handled inside the class"""
        # Adding moving averages and basic features
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        return self.data

    def load_data(self):
        """Return processed data"""
        return self.data
