# Main script (main.py)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from Green_Robo_Advisor_Class import RoboAdvisor  # Importing the class

# Step 1: Initialize and Load Data using the class
advisor = RoboAdvisor('Green ETF Selection.xlsx')

# Step 2: Preprocess and apply basic feature engineering using the class
data = advisor.preprocess_data()
data = advisor.feature_engineering()

# Step 3: Additional feature engineering in the main script
data['RSI_14'] = (100 - (100 / (1 + data['Close'].diff().rolling(14).mean() / 
                                 -data['Close'].diff().rolling(14).mean())))
data['Close_lag_1'] = data['Close'].shift(1)
data['Open_Close_Spread'] = data['Open'] - data['Close']
data['Cumulative_returns'] = (1 + data['Close'].pct_change()).cumprod() - 1

# Step 4: Remove rows with missing data caused by rolling windows and shifts
data = data.dropna()

# Step 5: Prepare the features and target
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI_14', 'Close_lag_1', 'Open_Close_Spread']].values
target = data['Close'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.reshape(-1, 1))

# Step 6: Create dataset for ANN
def create_dataset(data, target, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        Y.append(target[i + time_step])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_features, scaled_target, time_step)

# Step 7: Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Step 8: Build and Train the ANN model
model = Sequential()

# Add layers
model.add(Dense(units=256, input_dim=X_train.shape[1] * X_train.shape[2], activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Flatten the input for ANN (since we are not using LSTM)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Step 9: Make predictions and reverse scaling
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
Y_test_actual = scaler.inverse_transform(Y_test)

# Step 10: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(len(Y_test_actual)), Y_test_actual, color='blue', label='Actual Price')
plt.plot(range(len(predictions)), predictions, color='red', label='Predicted Price')
plt.title('ETF Price Prediction with ANN')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the model
model.save('ann_price_prediction_model_with_class.h5')
