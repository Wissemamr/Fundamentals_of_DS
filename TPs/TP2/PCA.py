import numpy as np
import pandas as pd

"""
NOTE :
PCA is a dimensionality reduction technique

This script contains the implementation of PCA from scratch and then applying dimensionality reduction
Major steps : 
1. Standardize the data
2. Calculate the covariance matrix
3. Calculate the eigenvalues and eigenvectors
4. Sort the eigenvalues and eigenvectors
5. Select the main components
6. Reduce the data

"""

DEBUG: bool = False


# --------------Utilities-----------------
def calculate_mean(column_data: list[float]) -> float:
    return round(np.mean(column_data), 2)


def calculate_std(column_data: list[float]) -> float:
    return round(np.std(column_data), 2)


def calculate_Z_score(column_data: list) -> list[float]:
    Z_score = (column_data - calculate_mean(column_data)) / calculate_std(column_data)
    return Z_score


def save_standardized_data(
    df: pd.DataFrame, convert_df_to_ndarray: bool = True
) -> pd.DataFrame | np.ndarray:
    df_stand = pd.DataFrame()
    for var in df.columns:
        df_stand[var] = calculate_Z_score(df[var])
    if convert_df_to_ndarray:
        return df_stand.to_numpy()
    else:
        return df_stand


def calculate_covariance(column_data: np.ndarray) -> np.ndarray:
    return np.cov(column_data, rowvar=False)


def get_eigen_val_vect(cov_mat: np.ndarray) -> np.ndarray:
    eigenvals, eigenvects = np.linalg.eig(cov_mat)
    return eigenvals, eigenvects


def sort_eigens(eigen_vals: np.ndarray, eigens_vects: np.ndarray) -> np.ndarray:
    """Sort eigenvalues and corresponding eigenvectors in descedning order"""
    sorted_ids = np.argsort(eigen_vals)[::-1]
    sorted_eigenvalues = eigen_vals[sorted_ids]
    sorted_eigenvectors = eigens_vects[:, sorted_ids]
    return sorted_eigenvalues, sorted_eigenvectors


def select_main_comp(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, threshold: float
) -> np.ndarray:
    """Select the main eigenvalues that satisfy the explained variance ratio and their
    corresponding eigenvectors"""
    total_variance = np.sum(eigenvalues)
    # calculate explained variance ratio (evr)
    evr = eigenvalues / total_variance
    cumulative_explained_variance = np.cumsum(evr)
    num_components_to_keep = np.argmax(cumulative_explained_variance >= threshold) + 1
    chosen_eigvals = eigenvalues[:num_components_to_keep]
    chosen_eigvects = eigenvectors[:, :num_components_to_keep]
    return chosen_eigvects


def reduce_data(stand_data: np.ndarray, principal_conponents: np.ndarray) -> np.ndarray:
    return np.dot(stand_data, principal_conponents)


def save_reduced_data(reduced_data: np.ndarray) -> pd.DataFrame:
    reduced_df = pd.DataFrame(
        reduced_data, columns=["Principal_Component_1", "Principal_Component_2"]
    )
    return reduced_df


# --------------MAIN PCA SCRIPT-----------------


def apply_PCA(unreduced_data: pd.DataFrame) -> pd.DataFrame:
    stand_data_ndarr = save_standardized_data(unreduced_data)
    if DEBUG:
        print(f"The standardized data with z-score : {stand_data_ndarr}")
    cov_matrix = calculate_covariance(stand_data_ndarr)
    eigenvals, eigenvects = get_eigen_val_vect(cov_matrix)
    sorted_eigenvals, sorted_eigenvects = sort_eigens(eigenvals, eigenvects)
    if DEBUG:
        print(
            f"The sorted eigenvalues: {sorted_eigenvals} \n The sorted eigenvects : {sorted_eigenvects}"
        )
    main_comp = select_main_comp(sorted_eigenvals, sorted_eigenvects, 0.9)
    reduced_data = reduce_data(stand_data_ndarr, main_comp)
    if DEBUG:
        print(reduce_data.shape)
    reduced_df = save_reduced_data(reduced_data=reduced_data)
    return reduced_df


# -----------------TESTING----------------------

if __name__ == "__main__":
    data = {
        "X": [12, 17, 12, 6, 17, 4],
        "Y": [24, 15.5, 13, 13.5, 21, 20.3],
        "Z": [6, -2, 3, -2.5, 7.2, -0.9],
    }

    df = pd.DataFrame(data, index=["R1", "R2", "R3", "R4", "R5", "R6"])
    result: pd.DataFrame = apply_PCA(unreduced_data=df)
    print(f"\nTHE NEW REDUCED DATAFRAME IS : \n {result}\n")
