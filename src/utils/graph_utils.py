import numpy as np
import torch
import scipy.sparse as sp

def normalize(mx):
    """
    Normalize sparse matrix.
    
    Args:
        mx: scipy.sparse matrix
    
    Returns:
        normalized sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    
    Args:
        adj: adjacency matrix in scipy.sparse format
    
    Returns:
        normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def calc_laplacian(adj, normalize_laplacian=True):
    """
    Calculate Laplacian of graph.
    
    Args:
        adj: adjacency matrix
        normalize_laplacian: whether to normalize Laplacian
    
    Returns:
        Laplacian matrix in scipy.sparse format
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    if not normalize_laplacian:
        # Unnormalized Laplacian: L = D - A
        d_mat = sp.diags(d.flatten())
        laplacian = d_mat - adj
    else:
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        laplacian = sp.eye(adj.shape[0]) - normalized_adj
    
    return laplacian

def calc_chebyshev_polynomials(laplacian, K):
    """
    Calculate Chebyshev polynomials for graph convolution.
    
    Args:
        laplacian: Laplacian matrix
        K: Order of Chebyshev polynomial
    
    Returns:
        List of Chebyshev polynomials of the Laplacian
    """
    # Rescale Laplacian to [-1, 1]
    laplacian = 2 * laplacian / laplacian.max() - sp.eye(laplacian.shape[0])
    
    # Initialize Chebyshev polynomials
    cheb_polynomials = [sp.eye(laplacian.shape[0]), laplacian]
    
    # Recursively compute higher order Chebyshev polynomials
    for k in range(2, K):
        cheb_polynomial = 2 * laplacian.dot(cheb_polynomials[k-1]) - cheb_polynomials[k-2]
        cheb_polynomials.append(cheb_polynomial)
    
    return cheb_polynomials

def prepare_graph_data(adjacency_matrix, graph_conv_type='cheb_graph_conv', K=3):
    """
    Prepare graph data for STGCN model.
    
    Args:
        adjacency_matrix: Room adjacency matrix
        graph_conv_type: Type of graph convolution ('cheb_graph_conv' or 'graph_conv')
        K: Order of Chebyshev polynomial
    
    Returns:
        GSO: Graph shift operator
    """
    # Make sure adjacency matrix is in scipy.sparse format
    if not sp.issparse(adjacency_matrix):
        adjacency_matrix = sp.coo_matrix(adjacency_matrix)
    
    # Add self-connections (diagonal)
    adj_with_self = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
    
    if graph_conv_type == 'cheb_graph_conv':
        # Calculate normalized Laplacian
        laplacian = calc_laplacian(adj_with_self, normalize_laplacian=True)
        
        # For Chebyshev polynomials, return the normalized Laplacian
        gso = laplacian.todense()
        return torch.FloatTensor(gso)
    else:
        # For simple graph convolution, return the normalized adjacency matrix
        normalized_adj = normalize_adj(adj_with_self)
        gso = normalized_adj.todense()
        return torch.FloatTensor(gso)
