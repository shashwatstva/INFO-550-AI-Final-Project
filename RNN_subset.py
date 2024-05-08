import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss, classification_report
import matplotlib.pyplot as plt


### Data Preprocessing
## Loading the dataset
data = pd.read_csv('subset.csv')

## Converting date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Handling missing values if any
data.dropna(inplace=True)

# Encode categorical variables (weather) using label encoding
label_encoder = LabelEncoder()
data['weather_encoded'] = label_encoder.fit_transform(data['weather'])

# Normalizing numerical variables
scaler = StandardScaler()
numerical_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

def split_data(dataset):
    # Split the dataset into input features and target variable
    X = dataset[['precipitation', 'temp_max', 'temp_min', 'wind']].values
    y = dataset['weather_encoded'].values  # Use encoded 'weather' column for labels

    return X, y

## RNN Specific
X, y = split_data(data)

# Reshape input features for RNN
X = X.reshape(-1, 1, 4)  # Reshape to (samples, time steps, features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of output classes
num_classes = len(np.unique(y))

# Define the RNN architecture for multi-class classification
model_rnn = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Start time measurement
start_time = time.time()

# Compile the model
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# End time measurement
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("RNN Execution Time:", execution_time)

# Evaluate the model
y_pred_proba_rnn = model_rnn.predict(X_test)
y_pred_classes_rnn = np.argmax(y_pred_proba_rnn, axis=1)

# Evaluate the model
accuracy_rnn = accuracy_score(y_test, y_pred_classes_rnn)
log_loss_rnn = log_loss(y_test, y_pred_proba_rnn)
classification_report_rnn = classification_report(y_test, y_pred_classes_rnn)

# Print evaluation metrics
print("RNN Model Evaluation:")
print(f"Accuracy: {accuracy_rnn}")
print(f"Log Loss: {log_loss_rnn}")
print("Classification Report:")
print(classification_report_rnn)

# Visualize the RNN architecture
layers = [layer.name for layer in model_rnn.layers]

plt.figure(figsize=(8, 6))
plt.title('RNN Architecture')
plt.xlabel('Layers')
plt.ylabel('Units')

for i, layer in enumerate(layers):
    plt.text(i, i, layer, fontsize=12, ha='center', va='center')
    plt.scatter(i, i, s=1000, color='skyblue', marker='o')

for i in range(len(layers) - 1):
    plt.plot([i, i+1], [i, i+1], linestyle='--', color='gray')

plt.grid(False)
plt.tight_layout()
plt.savefig('rnn_subset_architecture.png', format='png', dpi=900)
plt.show()

