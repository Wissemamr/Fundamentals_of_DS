import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from typing import Tuple


"""
Experimenting with K-NN
"""

DEBUG: bool = False


def clean_titanic_data(data_df_path: str) -> pd.DataFrame:
    # Load and clean the data
    df = pd.read_csv(data_df_path)
    if DEBUG:
        print(f"The raw df : \n {df.head()} \nOf shape {df.shape}")
    df.drop_duplicates(inplace=True)
    cleaned_df = df.drop(columns=["Cabin", "Name", "Ticket"])
    mean_age = cleaned_df["Age"].mean()
    cleaned_df["Age"].fillna(mean_age, inplace=True)
    cleaned_df.dropna(subset=["Embarked"], inplace=True)
    if DEBUG:
        print(f"The df shape after cleaning {cleaned_df.shape}")
    # STandardize the data
    scaler = StandardScaler()
    cleaned_df[["Age", "Fare"]] = scaler.fit_transform(cleaned_df[["Age", "Fare"]])
    return cleaned_df


def get_features_IG(cleaned_df: pd.DataFrame) -> pd.Series:
    # Calculate the information gain of each feature
    X = cleaned_df.drop("Survived", axis=1)  # get df without target column
    y = cleaned_df["Survived"]  # target feature
    # Encode the categorcial column  with label encoding : gender 0 : female , 1 : male
    label_encoder = LabelEncoder()
    X_encoded = X.apply(label_encoder.fit_transform)
    # Compute mutual information
    mutual_info = mutual_info_classif(X_encoded, y)
    feature_mutual_info = pd.Series(mutual_info, index=X.columns)
    return feature_mutual_info


def train_val_data_split(
    X_encoded: pd.DataFrame,
) -> Tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_precision_scores(
    feature_mutual_info: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    # Test different configurations of number of features, k values, and distance measures
    features_to_test = [2, 3, 4]
    k_values = [3, 5, 7]
    distance_measures = ["euclidean", "manhattan", "cosine"]

    results = []

    # try all possible combinations
    for num_features in features_to_test:
        for k in k_values:
            for distance_measure in distance_measures:
                # Select the top 'num_features' based on mutual information
                top_features = list(X.columns[np.argsort(mutual_info)[-num_features:]])

                # Train the K-NN classifier
                knn_classifier = KNeighborsClassifier(
                    n_neighbors=k, metric=distance_measure
                )
                knn_classifier.fit(X_train[top_features], y_train)

                # Make predictions on the test set
                y_pred = knn_classifier.predict(X_test[top_features])

                # Measure precision
                precision = precision_score(y_test, y_pred)
                # Store the results in a dict
                results.append(
                    {
                        "NumFeatures": num_features,
                        "K": k,
                        "DistanceMeasure": distance_measure,
                        "Precision": precision,
                    }
                )

    # Visualize and interpret results
    results_df = pd.DataFrame(results)
    return results_df


# get the com with maximum precision


def get_best_precision_combination(results_df: pd.DataFrame) -> pd.Series:
    best_combi_row = results_df.loc[results_df["Precision"].idxmax()]
    return best_combi_row



def main(data : pd.DataFrame)-> pd.Series:
    '''
    takes the titanic dataframe, cleans it, applies knn to predcit whether a passenger woul survive or not based on the passenger's data
    
    '''
    
    
