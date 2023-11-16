import numpy as np
from scipy.linalg import eigh

"""
NOTE This script contains the implementation of PCA from scratch and then applying dimensionality reduction
"""


#  Compute the Centered Matrix Mc
def compute_centered_matrix(M):
    n = M.shape[0]
    Cn = np.identity(n) - (1 / n) * np.ones((n, n))
    Mc = Cn.dot(M)
    return Mc


#  Reduce the Data
def reduce_data(Mc):
    n = Mc.shape[0]
    NIn = np.identity(n) / n
    VarM = Mc.T.dot(NIn).dot(Mc)
    VarV = np.diag(np.diag(VarM))
    IVarV = np.linalg.inv(VarV)
    Mr = Mc.dot(IVarV)
    return Mr


# Calculate the Covar Matrix
def calculate_covar_matrix(Mr):
    n_diag = np.diag(1 / np.sqrt(np.diag(Mr.T.dot(Mr))))
    Cov = Mr.T.dot(n_diag).dot(Mr)
    return Cov


# Calculate Eigenvalues and Eigenvectors
def calculate_eigenvalues_and_eigenvectors(Cov):
    eigenvalues, eigenvectors = eigh(Cov)
    return eigenvalues, eigenvectors


# Filter Eigenvectors
"""
NOTE We can select top k eigen vectors based on how much compression do we want.
The optimal way of selecting the number of components is to compute the explained var 
of each feature. We compute explained var by dividing the eigen values by the sum of 
all eigen values. Then, we take the cumulative sum of all eigen values."""


def filter_eigenvectors(eigenvalues, eigenvectors, var_thresh=0.9):
    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues / total_var
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    num_components = np.argmax(cumulative_var_ratio >= var_thresh) + 1
    top_eigenvectors = eigenvectors[:, :num_components]
    return top_eigenvectors


# Step 6: Project M into the New Dimensions
def project_into_new_dimensions(M, top_eigenvectors):
    projected_data = M.dot(top_eigenvectors)
    return projected_data


def apply_PCA (unreduced_data : np.ndarray)-> np.ndarray :
    
    Mc = compute_centered_matrix(M)
    Mr = reduce_data(Mc)
    Cov = calculate_covar_matrix(Mr)
    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(Cov)
    top_eigenvectors = filter_eigenvectors(eigenvalues, eigenvectors, var_thresh=0.9)
    projected_data = project_into_new_dimensions(M, top_eigenvectors)
    return projected_data
    
    

if __name__ == "__main__":
    M = np.array(
        [
            [12, 3, 6],
            [17, 13, -2],
            [12, 13, 3],
            [6, 13.5, -2.5],
            [17, 21, 7],
            [4, 20.5, -1],
        ],
        dtype=float,
    )

    result = apply_PCA(M)
    print(result)