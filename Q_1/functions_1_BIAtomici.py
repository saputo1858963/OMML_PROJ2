# Helper functions for Question 1 - Group BIAtomici

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import time

def extract_data():
    """
    Load and prepare FashionMNIST folder containing the training and testing data.
    Based on the provided extractor.py.
    """
    # Adjust path to step up one level to find FashionMNIST
    # Assuming structure: Project/Q_1/run.py -> need Project/FashionMNIST
    train_path = os.path.join('..', 'FashionMNIST', 'fashion-mnist_train.csv')
    test_path = os.path.join('..', 'FashionMNIST', 'fashion-mnist_test.csv')

    # Safety check for path (user specific)
    if not os.path.exists(train_path):
        # Fallback if running from root
        train_path = os.path.join('FashionMNIST', 'fashion-mnist_train.csv')
        test_path = os.path.join('FashionMNIST', 'fashion-mnist_test.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Extract labels 2 (Pullover) and 3 (Dress)
    indexes = [2, 3]
    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:, 1:]
    Y = train_set.values[:, 0]
    X_Test = test_set.values[:, 1:]
    Y_Test = test_set.values[:, 0]

    # Subset: 1000 samples per class for training
    ind_2 = np.where(Y == 2)[0][:1000]
    ind_3 = np.where(Y == 3)[0][:1000]
    
    # Subset: 200 samples per class for testing
    ind_2_test = np.where(Y_Test == 2)[0][:200]
    ind_3_test = np.where(Y_Test == 3)[0][:200]

    # Assign labels: +1 for Pullover (2), -1 for Dress (3)
    X2 = X[ind_2]; Y2 = np.ones(len(ind_2))
    X3 = X[ind_3]; Y3 = -np.ones(len(ind_3))
    
    X2t = X_Test[ind_2_test]; Y2t = np.ones(len(ind_2_test))
    X3t = X_Test[ind_3_test]; Y3t = -np.ones(len(ind_3_test))

    X_train = np.concatenate((X2, X3))
    Y_train = np.concatenate((Y2, Y3))
    X_test = np.concatenate((X2t, X3t))
    Y_test = np.concatenate((Y2t, Y3t))

    # Shuffle
    np.random.seed(1)
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # Scaling (Important for SVM!)
    # Simple Min-Max scaling to [0,1] or Standardization. 
    # Pixel values are 0-255, so dividing by 255 is standard.
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

def rbf_kernel(X1, X2, gamma):
    """
    Computes RBF kernel matrix: K(x, y) = exp(-gamma * ||x - y||^2)
    Uses vectorized Euclidean distance calculation.
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)
    dot_product = np.dot(X1, X2.T)
    dist_sq = sq_norm1 + sq_norm2 - 2 * dot_product
    return np.exp(-gamma * dist_sq)

def objective_function(alpha, Q):
    """
    Dual Objective: 0.5 * alpha^T * Q * alpha - sum(alpha)
    """
    return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)

def objective_gradient(alpha, Q):
    """
    Gradient of Dual Objective: Q * alpha - 1
    """
    return np.dot(Q, alpha) - np.ones_like(alpha)

def solve_svm_dual(X, y, C, gamma):
    """
    Solves the Dual SVM problem using Scipy SLSQP with progress tracking.
    """
    n_samples = X.shape[0]
    
    print(f" -> Computing Kernel Matrix for {n_samples} samples...")
    K = rbf_kernel(X, X, gamma)
    print(" -> Computing Q Matrix...")
    Q = np.outer(y, y) * K
    
    # Constraint: sum(alpha_i * y_i) = 0
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y}
    bounds = [(0, C) for _ in range(n_samples)]
    alpha0 = np.zeros(n_samples)
    
    # Progress callback function
    def progress_callback(xk):
        print(f"   [Optimization] Iteration completed. Current Objective: {objective_function(xk, Q):.4f}")

    print(" -> Starting Optimization (this may take time)...")
    start_time = time.time()
    
    # Reduced maxiter to 50 for testing speed, increase back to 100-200 for final result
    result = minimize(fun=objective_function, 
                      x0=alpha0, 
                      args=(Q,), 
                      method='SLSQP', 
                      jac=objective_gradient, 
                      bounds=bounds, 
                      constraints=constraints,
                      callback=progress_callback,
                      options={'maxiter': 50, 'ftol': 1e-4, 'disp': True}) 
                      
    end_time = time.time()
    print(" -> Optimization Finished.")
    
    alpha_opt = result.x
    
    # Compute Bias b
    sv_indices = np.where((alpha_opt > 1e-5) & (alpha_opt < C - 1e-5))[0]
    if len(sv_indices) > 0:
        b_values = []
        for idx in sv_indices:
            prediction_part = np.dot(alpha_opt * y, K[:, idx])
            b_values.append(y[idx] - prediction_part)
        b = np.mean(b_values)
    else:
        b = 0.0

    stats = {
        'time': end_time - start_time,
        'iterations': result.nit,
        'final_obj': result.fun,
        'status': result.message,
        'alpha': alpha_opt,
        'b': b
    }
    
    return alpha_opt, b, stats

def calculate_kkt_gap(alpha, Q, y, C):
    """
    Calculates the KKT violation gap: m(alpha) - M(alpha)
    Based on Keerthi/Gilbert definition for stopping criteria.
    gradient_i = (Q * alpha)_i - 1
    F_i = -y_i * gradient_i = y_i - (prediction_without_b)
    """
    # Gradient of objective f(alpha)
    grad = np.dot(Q, alpha) - 1.0
    
    # Optimality conditions related quantities
    # We want y_i * (model_output_i) >= 1
    # Let model_output_without_b = sum(alpha_j y_j K_ij)
    # Note: grad_i = y_i * model_output_without_b - 1
    # So y_i * grad_i = model_output_without_b - y_i
    
    # KKT Conditions checking sets:
    # I_up: Indices where alpha can increase
    # I_low: Indices where alpha can decrease
    
    I_up = np.where(((alpha < C - 1e-6) & (y == 1)) | ((alpha > 1e-6) & (y == -1)))[0]
    I_low = np.where(((alpha > 1e-6) & (y == 1)) | ((alpha < C - 1e-6) & (y == -1)))[0]

    # F_i = -grad f(alpha)_i / y_i  (approx, depends on formulation)
    # Let's stick to standard formulation:
    # r_i = y_i * f(x_i) - 1  (residual)
    # We use the gradient directly:
    # y_i * grad_i
    
    vals = y * grad
    
    # b_up = min_{i in I_up} (y_i * grad_i)
    # b_low = max_{i in I_low} (y_i * grad_i)
    # The gap is b_low - b_up. At optimality b_low <= b_up.
    
    b_up = np.min(vals[I_up]) if len(I_up) > 0 else 0
    b_low = np.max(vals[I_low]) if len(I_low) > 0 else 0
    
    return b_low - b_up

def predict(X_train, y_train, alpha, b, X_new, gamma):
    """
    y(x) = sign( sum(alpha_i * y_i * K(x_i, x)) + b )
    """
    K_new = rbf_kernel(X_train, X_new, gamma) # Shape (N_train, N_new)
    decision = np.dot(alpha * y_train, K_new) + b
    return np.sign(decision)

def compute_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred) * 100
    # Confusion Matrix (2x2)
    # Classes: 2 (mapped to +1), 3 (mapped to -1)
    # +1: Pullover, -1: Dress
    # Format: [[TP, FN], [FP, TN]] or similar. Let's do standard:
    # Row: True, Col: Pred
    # Labels: +1 (Pullover), -1 (Dress)
    
    # Map back to indices 0 and 1 for matrix
    # +1 -> index 0, -1 -> index 1
    y_t = np.where(y_true == 1, 0, 1)
    y_p = np.where(y_pred == 1, 0, 1)
    
    K = 2
    cm = np.zeros((K, K), dtype=int)
    for i in range(len(y_true)):
        cm[y_t[i], y_p[i]] += 1
        
    return accuracy, cm

def grid_search(X, y, k=5):
    """
    Simple Grid Search for C and Gamma using k-fold CV.
    """
    C_values = [0.1, 1, 10]
    gamma_values = [0.01, 0.1, 1]
    
    best_acc = 0
    best_params = (1, 0.1) # Default
    
    # Split indices for CV
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    fold_sizes = len(y) // k
    
    print("Starting Grid Search...")
    for C in C_values:
        for g in gamma_values:
            accuracies = []
            for i in range(k):
                val_idx = indices[i*fold_sizes : (i+1)*fold_sizes]
                train_idx = np.concatenate([indices[:i*fold_sizes], indices[(i+1)*fold_sizes:]])
                
                X_cv_train, y_cv_train = X[train_idx], y[train_idx]
                X_cv_val, y_cv_val = X[val_idx], y[val_idx]
                
                # Train (Simplified/Faster for CV - maybe fewer iters)
                # Note: Solving full SVM inside CV loop can be slow.
                # In real scenario, use small subset or fewer iters.
                alpha, b, _ = solve_svm_dual(X_cv_train, y_cv_train, C, g)
                
                y_val_pred = predict(X_cv_train, y_cv_train, alpha, b, X_cv_val, g)
                acc, _ = compute_metrics(y_cv_val, y_val_pred)
                accuracies.append(acc)
            
            avg_acc = np.mean(accuracies)
            # print(f"Checked C={C}, Gamma={g}, Avg Acc={avg_acc:.2f}%")
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = (C, g)
                
    return best_params