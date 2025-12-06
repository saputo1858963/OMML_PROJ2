# Helper functions for Question 2 - Group BIAtomici

import numpy as np
import pandas as pd
import os
import time
from scipy.optimize import minimize

def extract_data():
    """
    Load and prepare FashionMNIST folder containing the training and testing data.
    """
    train_path = os.path.join('..', 'FashionMNIST', 'fashion-mnist_train.csv')
    test_path = os.path.join('..', 'FashionMNIST', 'fashion-mnist_test.csv')

    if not os.path.exists(train_path):
        train_path = os.path.join('FashionMNIST', 'fashion-mnist_train.csv')
        test_path = os.path.join('FashionMNIST', 'fashion-mnist_test.csv')

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

    # Scale data
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

def get_kkt_violation_indices(alpha, grad, y, C):
    """
    Identify indices that violate KKT conditions using Keerthi/Gilbert definition.
    
    I_up:  Indices where we can move in direction +y_i (Increase F)
           {i | y_i = +1, alpha_i < C}  U  {i | y_i = -1, alpha_i > 0}
           
    I_low: Indices where we can move in direction -y_i (Decrease F)
           {i | y_i = +1, alpha_i > 0}  U  {i | y_i = -1, alpha_i < C}
    """
    
    # y_i * grad_i is the optimality condition metric
    yg = y * grad
    
    # Create masks for the sets
    # Set I_up
    mask_up_1 = (y == 1) & (alpha < C - 1e-6)
    mask_up_2 = (y == -1) & (alpha > 1e-6)
    I_up = np.where(mask_up_1 | mask_up_2)[0]
    
    # Set I_low
    mask_low_1 = (y == 1) & (alpha > 1e-6)
    mask_low_2 = (y == -1) & (alpha < C - 1e-6)
    I_low = np.where(mask_low_1 | mask_low_2)[0]
    
    return I_up, I_low, yg

def solve_subproblem(Q_sub, y_sub, linear_term_sub, C, current_alpha_sub):
    """
    Solves the small quadratic problem for the working set.
    """
    def sub_obj(alpha_q):
        # 0.5 * a' Q a + linear * a
        return 0.5 * np.dot(alpha_q, np.dot(Q_sub, alpha_q)) + np.dot(linear_term_sub, alpha_q)
    
    def sub_jac(alpha_q):
        return np.dot(Q_sub, alpha_q) + linear_term_sub

    # Equality constraint: sum(y_i * alpha_i) = - sum(y_fixed * alpha_fixed) = constant
    # The solver handles "constant" automatically if we enforce dot(alpha, y) = current_val
    target_sum = np.dot(current_alpha_sub, y_sub)
    
    constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y_sub) - target_sum, 'jac': lambda a: y_sub}
    bounds = [(0, C) for _ in range(len(y_sub))]
    
    res = minimize(sub_obj, current_alpha_sub, method='SLSQP', jac=sub_jac, 
                   bounds=bounds, constraints=constraints, options={'ftol': 1e-8, 'disp': False})
    return res.x

def solve_svm_decomposition(X, y, C, gamma, q=4, max_outer_iter=2000, tol=1e-3):
    """
    Decomposition method for Dual SVM.
    q: size of working set (must be even, >= 4)
    """
    n_samples = X.shape[0]
    
    # 1. Initialization
    alpha = np.zeros(n_samples)
    
    print(" -> Precomputing Kernel Matrix (this happens once)...")
    K = rbf_kernel(X, X, gamma)
    Q = np.outer(y, y) * K
    
    # Helper to compute full gradient: Q * alpha - 1
    # We update gradient iteratively or recompute (recompute is safer for stability here)
    
    start_time = time.time()
    
    print(f" -> Starting Decomposition Loop (q={q})...")
    
    iteration = 0
    final_obj = 0
    
    for it in range(max_outer_iter):
        iteration = it
        
        # 2. Compute Gradient of the dual: g = Q * alpha - 1
        grad = np.dot(Q, alpha) - 1.0
        
        # 3. Check KKT and Select Working Set
        I_up, I_low, yg = get_kkt_violation_indices(alpha, grad, y, C)
        
        # Filter valid indices for selection
        # We want to pick indices i from I_up with LARGEST yg
        # We want to pick indices j from I_low with SMALLEST yg
        # gap = max(yg[I_up]) - min(yg[I_low])
        
        # Sort I_up descending by yg
        I_up_sorted = I_up[np.argsort(yg[I_up])[::-1]]
        # Sort I_low ascending by yg
        I_low_sorted = I_low[np.argsort(yg[I_low])]
        
        if len(I_up_sorted) == 0 or len(I_low_sorted) == 0:
            break
            
        max_viol = yg[I_up_sorted[0]]
        min_viol = yg[I_low_sorted[0]]
        diff = max_viol - min_viol
        
        if diff < tol:
            print(f"   Converged at iter {it}. Gap: {diff:.6f}")
            break
            
        # Selection Rule: Pick q/2 from top of I_up and q/2 from top of I_low
        # This ensures we have pairs that can exchange mass to satisfy equality constraint
        n_select = q // 2
        
        ws_indices = np.concatenate([I_up_sorted[:n_select], I_low_sorted[:n_select]])
        ws_indices = np.unique(ws_indices) # Safety check
        
        # If we don't have enough candidates, just take what we have
        if len(ws_indices) < 2:
            break

        # 4. Construct Subproblem
        # We separate indices into Working (W) and Fixed (F)
        # alpha_W is variable, alpha_F is constant
        
        # Sub-Objective term: 0.5 * a_W' Q_WW a_W + (Q_WF * a_F - 1)_W * a_W
        # The linear term for the subproblem is: (Q[W, :] * alpha) - 1 - (Q[W, W] * alpha[W])
        # Which simplifies to: grad[W] - (Q[W, W] * alpha[W])
        # Wait, easier: standard form 0.5 a'Q a + c'x
        # c_sub = (Q_WF * alpha_F) - 1
        # c_sub = (Q_row_W * alpha) - (Q_WW * alpha_W) - 1
        # c_sub = grad[W] - (Q_WW * alpha_W)
        
        # Actually, let's just pass the full logic to the solver wrapper
        # The "linear part" that comes from fixed variables is:
        # sum_{j in Fixed} Q_ij * alpha_j - 1
        # We can extract this from the current full gradient:
        # linear_term_sub = grad[ws_indices] - np.dot(Q[np.ix_(ws_indices, ws_indices)], alpha[ws_indices])
        
        Q_sub = Q[np.ix_(ws_indices, ws_indices)]
        y_sub = y[ws_indices]
        current_alpha_sub = alpha[ws_indices]
        
        # The linear term for the QP subproblem
        # Derived from: 0.5*a_all*Q*a_all - sum(a_all)
        # relevant parts for a_sub: 0.5*a_sub*Q_sub*a_sub + a_sub * (Q_fixed*a_fixed - 1)
        linear_term_sub = grad[ws_indices] - np.dot(Q_sub, current_alpha_sub)
        
        # 5. Solve Subproblem
        new_alpha_sub = solve_subproblem(Q_sub, y_sub, linear_term_sub, C, current_alpha_sub)
        
        # 6. Update global alpha
        alpha[ws_indices] = new_alpha_sub
        
        if it % 100 == 0:
            obj_val = 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
            print(f"   Iter {it}: Obj={obj_val:.4f}, KKT Gap={diff:.6f}")

    end_time = time.time()
    
    # Calculate final stats
    final_obj = 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
    
    # Compute Bias b
    sv_indices = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
    if len(sv_indices) > 0:
        b_list = []
        for idx in sv_indices:
            # b = y_k - sum(alpha_j y_j K_jk)
            pred = np.dot(alpha * y, K[:, idx])
            b_list.append(y[idx] - pred)
        b = np.mean(b_list)
    else:
        b = 0.0
        
    stats = {
        'time': end_time - start_time,
        'iterations': iteration,
        'final_obj': final_obj,
        'status': 'Converged' if iteration < max_outer_iter else 'Max Iter Reached',
        'kkt_gap': diff if 'diff' in locals() else 0.0
    }
    
    return alpha, b, stats