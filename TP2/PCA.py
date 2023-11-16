import numpy as np
from scipy.linalg import eigh

"""
NOTE :
PCA provides a matrix of eigenvectors that explains the variance of the original
variables. To reduce data using PCA, we run the next operation: R = X V
Where, R is the reduced data (R has the shape of r data rows, and m
data columns), X is the original data matrix (i.e., it has the shape of r
data rows, and n columns). V is the eigenvectors matrix (it has the
shape of n data rows, and m data columns).


This script contains the implementation of PCA from scratch and then applying dimensionality reduction
Major steps : 

1. Calculate the reduction matrix V
2.Calculate the covariance matrix, C,
3.Find the eigenvalues of the cov matrix C
4.Sort the eigenvalues in descending order and arrange the
corresponding eigenvectors accordingly.
5.Decide how many principal components to keep based on the
explained variance
6. Form the projection matrix V by only keeping eigenvectors
corresponding to the highest eigenvalues.
7.Calculate the reduced data matrix R



"""

import numpy as np



#--------------Utilities-----------------
def standardize_data(data : np.ndarray) -> np.ndarray:
    '''Calculate the mean adn std and standardize the data using Z-score'''
    mean = np.mean(data, axis =0)
    std_dev = np.std(data, axis=0)
    standaridized_data = (data - mean) / std_dev
    return standardize_data

def get_cov_matrix(data : np.ndarray)-> np.ndarray:
    return np.cov(data, rowvar=False)

def get_eigen_val_vect(cov_mat : np.ndarray)-> tuple(np.ndarray):
    eigenvals, eigenvects = np.linalg.eig(cov_mat)
    return tuple(eigenvals, eigenvects)

def sort_eigens(eigen_vals : np.ndarray, eigens_vects : np.ndarray) -> tuple(np.ndarray):
    ''' Sort eigenvalues and corresponding eigenvectors'''
    sorted_ids= np.argsort(eigen_vals)[::-1]
    eigenvalues = eigenvalues[sorted_ids]
    eigenvectors = eigens_vects[:, sorted_ids]
    return eigenvalues , eigenvectors


    



def apply_PCA(unreduced_data:np.ndarray) -> np.ndarray:
    

    mean_values, std_dev_values = standardize_data(unreduced_data)

    # Step 2: 
    standardized_data = 

    # Step 3: Calculate the covariance matrix
    covariance_matrix = np.cov(standardized_data, rowvar=False)

    # Step 4: Find eigenvalues and eigenvectors
    #eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 5: Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 6: Decide on the number of principal components to keep
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    num_components_to_keep = np.argmax(cumulative_explained_variance >= 0.9) + 1

    # Step 7: Form the projection matrix V
    projection_matrix = eigenvectors[:, :num_components_to_keep]

    # Step 8: Calculate the reduced data matrix R
    reduced_data = np.dot(standardized_data, projection_matrix)



if __name__=='__main__':
    data = np.array([
    [12, 24, 6],
    [17, 15.5, -2],
    [12, 13, 3],
    [6, 13.5, -2.5],
    [17, 21, 7.2],
    [4, 20.3, -0.9]
])
    
    
    
'''

    print("Original Data:")
    print(data)
    print("\nStandardized Data:")
    print(standardized_data)
    print("\nCovariance Matrix:")
    print(covariance_matrix)
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)
    print("\nExplained Variance Ratio:")
    print(explained_variance_ratio)
    print("\nCumulative Explained Variance:")
    print(cumulative_explained_variance)
    print("\nNumber of Components to Keep:")
    print(num_components_to_keep)
    print("\nProjection Matrix:")
    print(projection_matrix)
    print("\nReduced Data:")
    print(reduced_data)   
'''