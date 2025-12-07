# Helper functions for Question 3 - Group BIAtomici

import numpy as np
import pandas as pd
import os
import time

def extract_data():
    """
    Load and prepare FashionMNIST folder using robust absolute paths.
    """
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

def solve_svm_mvp(X, y, C, gamma, max_iter=100000, tol=1e-3):
    """
    Solves Dual SVM using MVP (Most Violating Pair) Decomposition (q=2).
    """
    n_samples = X.shape[0]
    
    print(" -> Precomputing Kernel Matrix...")
    K = rbf_kernel(X, X, gamma)
    Q = np.outer(y, y) * K
    
    alpha = np.zeros(n_samples)
    
    # Gradient of dual objective (Minimization): grad = Q*alpha - 1
    # Initially alpha=0, so grad = -1
    grad = -np.ones(n_samples)
    
    start_time = time.time()
    print(" -> Starting MVP Optimization Loop...")
    
    iters = 0
    final_diff = 0
    
    for it in range(max_iter):
        iters = it
        
        # --- Step 1: Select MVP (Most Violating Pair) ---
        # yg = y_i * grad_i
        yg = y * grad
        
        # I_up: Indices where alpha can increase (y=1 & a<C) OR (y=-1 & a>0)
        mask_up = ((y == 1) & (alpha < C - 1e-6)) | ((y == -1) & (alpha > 1e-6))
        # I_low: Indices where alpha can decrease (y=1 & a>0) OR (y=-1 & a<C)
        mask_low = ((y == 1) & (alpha > 1e-6)) | ((y == -1) & (alpha < C - 1e-6))
        
        idx_up = np.where(mask_up)[0]
        idx_low = np.where(mask_low)[0]
        
        if len(idx_up) == 0 or len(idx_low) == 0:
            break
        
        # Keerthi/Gilbert Selection:
        # i (from I_up) should have MINIMAL yg (b_up)
        # j (from I_low) should have MAXIMAL yg (b_low)
        i = idx_up[np.argmin(yg[idx_up])]
        j = idx_low[np.argmax(yg[idx_low])]
        
        # KKT Gap = b_low - b_up
        diff = yg[j] - yg[i]
        final_diff = diff
        
        if diff < tol:
            print(f"   Converged at iter {it}. Gap: {diff:.6f}")
            break
            
        # --- Step 2: Analytic Solution ---
        a_i_old = alpha[i]
        a_j_old = alpha[j]
        y_i = y[i]
        y_j = y[j]
        
        # Compute bounds L and H
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
        
        # Update formula: a_j_new = a_j_old + y_j * (diff / eta)
        # Note: diff > 0, eta < 0. Term is negative.
        # This reduces alpha_j (if y_j=1) which is correct for "I_low"
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
        
        # --- Step 3: Update Gradient ---
        delta_i = a_i_new - a_i_old
        delta_j = a_j_new - a_j_old
        grad += delta_i * Q[:, i] + delta_j * Q[:, j]
        
        alpha[i] = a_i_new
        alpha[j] = a_j_new
        
        if it % 5000 == 0 and it > 0:
             print(f"   Iter {it}: Gap {diff:.5f}")

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