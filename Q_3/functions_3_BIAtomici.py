import numpy as np
import pandas as pd
import os
import time

def extract_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_path = os.path.join(project_root, 'FashionMNIST', 'fashion-mnist_train.csv')
    test_path = os.path.join(project_root, 'FashionMNIST', 'fashion-mnist_test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data not found at {train_path}. Check your folder structure.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    indexes = [2, 3]
    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:, 1:]
    Y = train_set.values[:, 0]
    X_Test = test_set.values[:, 1:]
    Y_Test = test_set.values[:, 0]

    # Subset 1000 for training, 200 for testing
    ind_2 = np.where(Y == 2)[0][:1000]
    ind_3 = np.where(Y == 3)[0][:1000]
    ind_2_test = np.where(Y_Test == 2)[0][:200]
    ind_3_test = np.where(Y_Test == 3)[0][:200]

    X2 = X[ind_2]; Y2 = np.ones(len(ind_2))
    X3 = X[ind_3]; Y3 = -np.ones(len(ind_3))
    
    X2t = X_Test[ind_2_test]; Y2t = np.ones(len(ind_2_test))
    X3t = X_Test[ind_3_test]; Y3t = -np.ones(len(ind_3_test))

    X_train = np.concatenate((X2, X3))
    Y_train = np.concatenate((Y2, Y3))
    X_test = np.concatenate((X2t, X3t))
    Y_test = np.concatenate((Y2t, Y3t))

    np.random.seed(1)
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

def rbf_kernel(X1, X2, gamma):
    sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)
    dot_product = np.dot(X1, X2.T)
    dist_sq = sq_norm1 + sq_norm2 - 2 * dot_product
    return np.exp(-gamma * dist_sq)

def predict(X_train, y_train, alpha, b, X_new, gamma):
    K_new = rbf_kernel(X_train, X_new, gamma)
    decision = np.dot(alpha * y_train, K_new) + b
    return np.sign(decision)

def compute_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred) * 100
    y_t = np.where(y_true == 1, 0, 1)
    y_p = np.where(y_pred == 1, 0, 1)
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        cm[y_t[i], y_p[i]] += 1
    return accuracy, cm

# Solve dual SVM with MVP
def solve_svm_mvp(X, y, C, gamma, max_iter=100000, tol=1e-3):
    n_samples = X.shape[0]
    
    K = rbf_kernel(X, X, gamma)
    Q = np.outer(y, y) * K
    
    alpha = np.zeros(n_samples)
    
    # Gradient of dual objective (minimization): grad = Q*alpha - 1
    # Initially alpha=0, so grad = -1
    grad = -np.ones(n_samples)
    
    start_time = time.time()
    
    iters = 0
    final_diff = 0
    
    for it in range(max_iter):
        iters = it
        
        # Select MVP (Most Violating Pair): the q = 2 variables that will be updated
        yg = y * grad  
        
        R = ((y == 1) & (alpha < C - 1e-6)) | ((y == -1) & (alpha > 1e-6))
        S = ((y == 1) & (alpha > 1e-6)) | ((y == -1) & (alpha < C - 1e-6))
        
        idx_up = np.where(R)[0]
        idx_low = np.where(S)[0]
        
        if len(idx_up) == 0 or len(idx_low) == 0:
            break
        
        i = idx_up[np.argmin(yg[idx_up])]
        j = idx_low[np.argmax(yg[idx_low])]
        
        # KKT Gap = b_low - b_up
        diff = yg[j] - yg[i]
        final_diff = diff
        
        if diff < tol:  # we are sufficiently optimum
            break
            
        # Analytic solution
        a_i_old = alpha[i]
        a_j_old = alpha[j]
        y_i = y[i]
        y_j = y[j]
        
        # Compute the admissible interval [L,H] of alpha_j_new
        if y_i != y_j:
            L = max(0, a_j_old - a_i_old)
            H = min(C, C + a_j_old - a_i_old)
        else:
            L = max(0, a_i_old + a_j_old - C)
            H = min(C, a_i_old + a_j_old)
            
        if L == H:
            continue
            
        # Curvature eta = 2*K_ij - K_ii - K_jj (Usually negative)
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        
        if eta >= -1e-12:
            continue
        
        # Update a_j
        a_j_new = a_j_old + (y_j * diff) / eta
        
        # Clip
        if a_j_new > H:
            a_j_new = H
        elif a_j_new < L:
            a_j_new = L
            
        if abs(a_j_new - a_j_old) < 1e-8:
            continue
            
        # Update alpha_i
        a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)
        
        # Update gradient 
        delta_i = a_i_new - a_i_old
        delta_j = a_j_new - a_j_old
        grad += delta_i * Q[:, i] + delta_j * Q[:, j]
        
        alpha[i] = a_i_new
        alpha[j] = a_j_new
        
    end_time = time.time()
    
    final_obj = 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
    
    # Compute Bias
    sv_indices = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
    if len(sv_indices) > 0:
        # b = -average(y * grad) for SVs
        b = -np.mean(y[sv_indices] * grad[sv_indices])
    else:
        b = 0.0

    stats = {
        'time': end_time - start_time,
        'iterations': iters,
        'final_obj': final_obj,
        'gap': final_diff,
        'status': 'Converged' if iters < max_iter else 'Max Iter',
        'cols_computed': n_samples
    }
    
    return alpha, b, stats