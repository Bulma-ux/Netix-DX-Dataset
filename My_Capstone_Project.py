import pandas as pd
import numpy as np

df = pd.read_excel("NETIX_DX_karnitin_dataset.xlsx")

df

df = df.iloc[:, :24]
df

# Function to detect and remove outliers using IQR method
def remove_outliers_iqr(df, k=1.5):
    df_no_outliers = pd.DataFrame()
    for column_name in df.columns[1:]:  # Start from the second column to skip the non-numeric column
        if df[column_name].dtype != 'object':  # Check if the column contains numeric data
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_no_outliers[column_name] = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)][column_name]
        else:
            df_no_outliers[column_name] = df[column_name]  # For non-numeric columns, keep the original data
    return df_no_outliers

# Call the function to remove outliers using IQR method
df_without_outliers = remove_outliers_iqr(df)

# Display the DataFrame without outliers
print(df_without_outliers)

# Orijinal veri çerçevesini güncelle
df.update(df_without_outliers)
#Bu işlem sonrasında df veri çerçevesi,
#outlier değerleri NaN ile doldurulmuş df_without_outliers veri çerçevesiyle güncellenecektir.
#Artık df veri çerçevesinde outlier değerlerini görmeyeceksiniz.

df

from sklearn.impute import KNNImputer

# Select all numerical values
numeric_columns = df.select_dtypes(include=[np.number])

# Initialize KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)

# Perform KNN imputation
data_imputed = knn_imputer.fit_transform(numeric_columns)

df_imputed = pd.DataFrame(data_imputed, columns=numeric_columns.columns)

df.update(df_imputed)

print(df)

#CHECK IF THERE ARE NAN VALUES IN DF

# Check for NaN values in the DataFrame
if df.isnull().any().any():
    print("There are NaN values in the DataFrame.")
else:
    print("There are no NaN values in the DataFrame.")

row_61 = df.loc[61]
row_61

import seaborn as sns
sns.heatmap(df.corr());

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Step 1: Prepare the Data
# Assuming you have already read the data into the DataFrame df
X = df.drop(columns=['HastaGrubu'])  # Drop the 'HastaGrubu' column for clustering
#X = X.fillna(X.mean())  # Fill any remaining NaN values with the mean of the column (not necessary at this point)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize the data

# Step 2: Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Step 3: Calculate Distances
# Distances between the control group (G0) and other clusters
control_group_index = np.where(df['HastaGrubu'] == 'G0')[0][0]
distances_to_other_clusters = pairwise_distances(X_scaled[control_group_index].reshape(1, -1), X_scaled, metric='euclidean')

# Distances between all clusters
cluster_distances = pairwise_distances(X_scaled, metric='euclidean')

# Step 4: Train a Classification Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you want to predict the 'HastaGrubu' column
y = df['HastaGrubu']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)

# To predict the group of a random patient, you can use the classifier as follows:
# Assuming you have a random_patient_values list containing the values of the features for the random patient
random_patient_values = [13.8 ,	1.0 ,	100.0 ,	3.0 ,	22.6 ,	10.4 ,	230.0 ,	109.7 ,	47.3 , 47.3 , 4.7, 6.83, 5 , 0.0 ,	0.2 	,-2.62 ,	-3.17 ,	0.0 ,	0.7 ,	42.42 ,	14.39 ,	26.4, 26.4]
random_patient_values_scaled = scaler.transform([random_patient_values])
predicted_group = clf.predict(random_patient_values_scaled)
print("Predicted Group:", predicted_group[0])

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Calculate Cluster Distances
cluster_distances = pairwise_distances(kmeans.cluster_centers_, metric='euclidean')

# Step 3: Create Clustermap
cluster_labels = [f"G{i+1}" for i in range(5)]
cluster_distance_df = pd.DataFrame(cluster_distances, index=cluster_labels, columns=cluster_labels)

plt.figure(figsize=(6, 4))
sns.clustermap(cluster_distance_df, annot=True, cmap="viridis", fmt=".2f", linewidths=.5)
plt.title("Cluster Distances")
plt.show()

#Create a DataFrame to store the mapping between cluster labels and HastaGrubu
cluster_mapping_df = pd.DataFrame({'Cluster': np.unique(cluster_labels), 'HastaGrubu': np.unique(y)})

# Sort the DataFrame based on cluster labels
cluster_mapping_df.sort_values(by='Cluster', inplace=True)

# Display the mapping between cluster labels and HastaGrubu
print(cluster_mapping_df)

"""To determine if the "G0" control group is located in the middle of the other clusters, you can use the cluster centroids obtained from the K-means clustering. The cluster centroids represent the average values of each feature for the data points within the cluster. By comparing the centroid of the "G0" cluster with the centroids of the other clusters, you can get an idea of how it is positioned relative to the other clusters."""

# Step 3: Get Cluster Centroids
cluster_centroids = kmeans.cluster_centers_

# Step 4: Plot the Cluster Centroids
plt.figure(figsize=(10, 8))
sns.scatterplot(data=cluster_centroids, x=cluster_centroids[:, 0], y=cluster_centroids[:, 1], hue=[f'G{i+1}' for i in range(5)], s=100, marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Cluster Centroids')
plt.legend()
plt.show()

"""G5 (corresponding to the control group, G0) is far from the centroids of the other clusters (representing different non-communicable metabolic disease groups, G1, G2, G3, G4) indicates that there are distinct differences between the control group and the other disease groups in terms of the feature values."""

# Create a dictionary with the reference values
reference_data = {
    'Değerin Tanımı': ['GLUKOZ', 'ürik asit', 'AST', 'ALT', 'total kolesterol', 'LDL-kolesterol', 'trigliserit', 'albumin', 'insulin', 'HbA1c', 'protein %', 'KH %', 'yağ %'],
    'Min': [70, 0, 5, 5, 130, 100, 0, 3.2, 2.6, 4.5, 10, 55, 25],
    'Max': [120, 50, 42, 45, 200, 130, 150, 5.5, 24.9, 5.7, 15, 65, 35],
    'Mean': [95, 25, 23.5, 25, 165, 115, 75, 4.35, 13.75, 5.1, 12.5, 60, 30]
}

# Create the DataFrame with the reference data
reference_df = pd.DataFrame(reference_data)

# Normalize the values to the range [-1, 0, 1]
for col in ['Min', 'Max', 'Mean']:
    reference_df[col] = 2 * (reference_df[col] - reference_df[col].min()) / (reference_df[col].max() - reference_df[col].min()) - 1

print(reference_df)

[120,	50,	42,	45,	200,	130,	150,	5.5,	24.9,	5.7,	15,	65,	35, 0.0 ,	0.2 	,-2.62 ,	-3.17 ,	0.0 ,	0.7 ,	42.42 ,	14.39 ,	26.4, 26.4]

"""To compare the old measurements with the new measurements and determine whether the patient has made progress or regressed, we can use the normalized reference table that we created earlier. Follow these steps:

    Normalize both the old and new measurements using the same normalization formula we used for the reference table.
    Calculate the Euclidean distance between the normalized old and new measurements.
    If the Euclidean distance is smaller than a predefined threshold, you can consider it as "no significant change" or "stable."
    If the Euclidean distance is larger than the threshold, you can consider it as "progress" or "regression."



"""

# Old and new measurements (replace with the actual values)
old_measurements = [ 95, 25, 23.5, 25, 165, 115, 75, 4.35, 13.75, 5.1, 12.5, 60, 30] # These are the mean values from the reference tables
new_measurements = [70, 0, 5, 5, 130, 100, 0, 3.2, 2.6, 4.5, 10, 5, 25] # These are the min values from the reference table

# Calculate percentage changes for each measurement
percentage_changes = [(new_val - old_val) / old_val * 100 for new_val, old_val in zip(new_measurements, old_measurements)]

# Set a threshold for significant change (you can adjust this based on your specific application)
threshold = 10.0  # 10% change as an example

# Determine if there is progress, regression, or no significant change
progress = any(change >= threshold for change in percentage_changes)
regression = any(change <= -threshold for change in percentage_changes)
if progress:
    print("Progress.")
elif regression:
    print("Regression.")
else:
    print("No significant change (stable).")

# Final Version without divison by zero error

# Old and new measurements (replace with the actual values)
old_measurements = [ 95, 25, 23.5, 25, 165, 115, 75, 4.35, 13.75, 5.1, 12.5, 60, 30]  # These are the mean values from the reference tables
new_measurements = [70, 0, 5, 5, 130, 100, 0, 3.2, 2.6, 4.5, 10, 5, 25]   # These are the min values from the reference table

""" According to the given old and new measurements, we expect to see regression"""

# Calculate percentage changes for each measurement
percentage_changes = []
for new_val, old_val in zip(new_measurements, old_measurements):
    if old_val == 0:
        percentage_changes.append(0)
    else:
        percentage_changes.append((new_val - old_val) / old_val * 100)

# Set a threshold for significant change (you can adjust this based on your specific application)
threshold = 10.0  # 10% change as an example

# Determine if there is progress, regression, or no significant change
progress = any(change >= threshold for change in percentage_changes)
regression = any(change <= -threshold for change in percentage_changes)
if progress:
    print("Progress.")
elif regression:
    print("Regression.")
else:
    print("No significant change (stable).")

# Normalize the old and new measurements using the same normalization formula as in the reference_df
normalized_old = 2 * (np.array(old_measurements) - reference_df['Min']) / (reference_df['Max'] - reference_df['Min']) - 1
normalized_new = 2 * (np.array(new_measurements) - reference_df['Min']) / (reference_df['Max'] - reference_df['Min']) - 1

# Create a DataFrame for the normalized old and new measurements
df_normalized = pd.DataFrame({'Değerin Tanımı': reference_df['Değerin Tanımı'], 'Önceden Alınan': normalized_old, 'Sonradan Alınan': normalized_new})

# Melt the DataFrame to convert it to long format for plotting
df_melted = df_normalized.melt(id_vars='Değerin Tanımı', var_name='Ölçüm Zamanı', value_name='Normalized Değer')

# Plot the hatplot for each measurement
plt.figure(figsize=(12, 6))
sns.pointplot(data=df_melted, x='Değerin Tanımı', y='Normalized Değer', hue='Ölçüm Zamanı', join=False, ci='sd', palette='Set2')
plt.xticks(rotation=90)
plt.ylabel('Normalize Değer')
plt.title('Hastanın Ölçüm Değerlerinin Gelişimi')
plt.show()

from scipy.stats import f_oneway

# Assuming you have already read the data into the DataFrame df
# You can also perform necessary data preparation steps (e.g., filling missing values) here if needed

# Filter the data for g0 control group and the other 4 HastaGrubu
control_group = df[df['HastaGrubu'] == 'G0']
other_groups = df[df['HastaGrubu'].isin(['G1', 'G2', 'G3', 'G4'])]

# Perform ANOVA test for each feature
anova_results = {}
for feature in df.columns.drop('HastaGrubu'):
    anova_result = f_oneway(control_group[feature], other_groups[feature])
    anova_results[feature] = {
        'F-Statistic': anova_result.statistic,
        'p-value': anova_result.pvalue
    }

# Create a DataFrame to display the ANOVA results
anova_df = pd.DataFrame.from_dict(anova_results, orient='index')

# Add a column to indicate the significance (You can adjust the significance level as needed)
alpha = 0.05
anova_df['Significant'] = anova_df['p-value'] < alpha

print(anova_df)

import seaborn as sns
import matplotlib.pyplot as plt

# Çizdirilecek görselleştirme yöntemi olarak pairplot kullanma
selected_features = ['AST', 'ALT', 'HOMA-IR DEĞERLENDİRMESİ', 'boy z skoru', 'ağırlık z skoru', 'whr', 'ağırlık']
sns.pairplot(df, hue='HastaGrubu', vars=selected_features, palette='Set1', markers='o')

# Grafiklerin daha net görünmesi için yatayda birbirlerinden boşluk bırakma
plt.subplots_adjust(wspace=0.5)

plt.show()

