import time
# import resource
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
import matplotlib.pyplot as plt
# import networkx as nx

# Step a: Data Preprocessing
# Load the dataset
data = pd.read_csv('subset.csv')

# Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Handle missing values if any
data.dropna(inplace=True)

# Step b: Model Construction
start_time = time.time()


# Define the structure of the Bayesian network
model_bayesian = BayesianNetwork([('precipitation','weather'),('temp_max', 'weather'),
                                  ('temp_min', 'weather'), ('wind', 'weather')])

end_time = time.time()

# Measure time complexity
time_complexity = end_time - start_time
print("Time Complexity:", time_complexity)

# Step c: Model Training
# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

start_time = time.time()
# Learn the parameters of the Bayesian network using Maximum Likelihood Estimation
model_bayesian.fit(train_data, estimator=MaximumLikelihoodEstimator)
end_time = time.time()

time_complexity = end_time - start_time
print("Time Complexity for Training:", time_complexity)



# Step d: Prediction
# Extract only the columns that are present in the model structure
test_data_filtered = test_data[['precipitation', 'temp_max', 'temp_min', 'wind']]
# Convert the DataFrame to a dictionary of evidence
evidence_dict = test_data_filtered.iloc[0].to_dict()

# Use probabilistic inference to make predictions
infer = VariableElimination(model_bayesian)
start_time = time.time()

predicted_weather_bayesian = infer.query(variables=['weather'], evidence=evidence_dict)
end_time = time.time()
print(predicted_weather_bayesian)

# Measure time complexity
time_complexity = end_time - start_time
print("Time Complexity for Prediction:", time_complexity)


# Extract predicted class probabilities
predicted_array_bayesian = predicted_weather_bayesian.values.flatten()

# Get the index of the predicted class with the highest probability
predicted_class_index_bayesian = predicted_array_bayesian.argmax()

# Map the index to the actual class label
predicted_class_bayesian = predicted_weather_bayesian.state_names['weather'][predicted_class_index_bayesian]

# Extract actual labels from the test data
actual_labels = test_data['weather']

# Repeat the predicted class for the length of actual_labels
predicted_classes_bayesian = [predicted_class_bayesian] * len(actual_labels)

# Evaluate Bayesian Network model
accuracy_bayesian = accuracy_score(actual_labels, predicted_classes_bayesian)
log_loss_bayesian = log_loss(actual_labels, [predicted_array_bayesian] * len(actual_labels))
classification_report_bayesian = classification_report(actual_labels, predicted_classes_bayesian)

# Print evaluation metrics
print("Bayesian Network Model Evaluation:")
print(f"Accuracy: {accuracy_bayesian}")
print(f"Log Loss: {log_loss_bayesian}")
print("Classification Report:")
print(classification_report_bayesian)

# Visualize the graph
# Create an empty directed graph
# nx_graph = nx.DiGraph()
# for node in model_bayesian.nodes():
#     nx_graph.add_node(node)
#
# # Add edges to the graph
# for edge in model_bayesian.edges():
#     nx_graph.add_edge(*edge)
#
# # Plot the NetworkX graph
# plt.figure(figsize=(10 ,10))
# pos = nx.spring_layout(nx_graph)  # Position nodes using Fruchterman-Reingold force-directed algorithm
# nx.draw(nx_graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
# plt.title("Bayesian Network")
# plt.savefig('bn_subset_graph.png', format='png', dpi=900)
# plt.show()
