import numpy as np
import pandas as pd
import os
import time
from scipy.optimize import minimize

def extract_data():
    """
    Load and prepare FashionMNIST folder using absolute paths for robustness.
    """
    # Robust path finding
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_path = os.path.join(project_root, 'FashionMNIST', 'fashion-mnist_train.csv')
    test_path = os.path.join(project_root, 'FashionMNIST', 'fashion-mnist_test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data not found at {train_path}. Check your folder structure.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Filter for labels 2 (Pullover) and 3 (Dress)
    indexes = [2, 3]
    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:, 1:]
    Y = train_set.values[:, 0]
    X_Test = test_set.values[:, 1:]
    Y_Test = test_set.values[:, 0]

    # Subset: 1000 for training, 200 for testing
    ind_2 = np.where(Y == 2)[0][:1000]
    ind_3 = np.where(Y == 3)[0][:1000]
    ind_2_test = np.where(Y_Test == 2)[0][:200]
    ind_3_test = np.where(Y_Test == 3)[0][:200]

    # Map labels: 2 -> +1, 3 -> -1
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

    # Scale to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

def rbf_kernel(X1, X2, gamma):
    """
    Computes RBF kernel matrix.
    """
    sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)
    dot_product = np.dot(X1, X2.T)
    dist_sq = sq_norm1 + sq_norm2 - 2 * dot_product
    return np.exp(-gamma * dist_sq)

def predict(X_train, y_train, alpha, b, X_new, gamma):
    """
    Computes predictions on new data.
    """
    K_new = rbf_kernel(X_train, X_new, gamma)
    decision = np.dot(alpha * y_train, K_new) + b
    return np.sign(decision)

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy and confusion matrix.
    """
    accuracy = np.mean(y_true == y_pred) * 100
    
    # Map +1 -> 0, -1 -> 1 for matrix indexing
    y_t = np.where(y_true == 1, 0, 1)
    y_p = np.where(y_pred == 1, 0, 1)
    
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        cm[y_t[i], y_p[i]] += 1
        
    return accuracy, cm

# Selection of the working set (dimensio q): it contains the most violating indices of the KKT
def get_working_set_indices(alpha, grad, y, C, q):
    """
    Selects q indices based on KKT violations.
    LOGIC: Picks q/2 indices from R with SMALLEST y*grad
           and q/2 indices from S with LARGEST y*grad.
    """
    # yg = y_i * grad_i where grad = Q * alpha - 1
    yg = y * grad
    
    # I_up: Indices where alpha can increase (y=+1 & a<C) OR (y=-1 & a>0)
    R = np.where(((y == 1) & (alpha < C - 1e-6)) | 
                    ((y == -1) & (alpha > 1e-6)))[0]
    
    # I_low: Indices where alpha can decrease (y=+1 & a>0) OR (y=-1 & a<C)
    S = np.where(((y == 1) & (alpha > 1e-6)) | 
                     ((y == -1) & (alpha < C - 1e-6)))[0]

    if len(R) == 0 or len(S) == 0:
        return np.array([]), 0.0

    # --- FIX: SWAPPED SORTING ORDER ---
    # We want min(yg) from I_up
    R_sorted = R[np.argsort(yg[R])] # Ascending (Smallest first)
    
    # We want max(yg) from I_low
    S_sorted = S[np.argsort(yg[S])[::-1]] # Descending (Largest first)

    # Calculate current Gap: max(I_low) - min(I_up)
    # Ideally should be positive if violation exists
    m = yg[S_sorted[0]]
    M = yg[R_sorted[0]]
    gap = m - M

    # select q/2 from each
    n_select = q // 2  # integer division

    # take q/2 indices from each set
    ws_indices = np.concatenate([R_sorted[:n_select], S_sorted[:n_select]])
    
    return np.unique(ws_indices), gap

# Resolution of the quadratic subproblem
def solve_subproblem(Q_sub, y_sub, linear_term, C, alpha_init):
    """
    Solves the small QP for the working set using SLSQP.
    """
    # objective function
    def sub_obj(a):
        return 0.5 * np.dot(a, np.dot(Q_sub, a)) + np.dot(linear_term, a)

    # jacobian
    def sub_jac(a):
        return np.dot(Q_sub, a) + linear_term

    # Constraint: sum(y_i * alpha_i) = constant
    current_eq_val = np.dot(alpha_init, y_sub)
    constraints = {'type': 'eq', 
                   'fun': lambda a: np.dot(a, y_sub) - current_eq_val, 
                   'jac': lambda a: y_sub}
    
    bounds = [(0, C) for _ in range(len(alpha_init))]  # 0 <= alpha <= C 
    
    res = minimize(sub_obj, alpha_init, jac=sub_jac, method='SLSQP', 
                   bounds=bounds, constraints=constraints, 
                   options={'ftol': 1e-8, 'disp': False})
    return res.x, res.nfev

# Decomposition framework
def solve_decomposition(X, y, C, gamma, q, max_iter=2000, tol=1e-3):
    """
    Main Decomposition Solver.
    """
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    
    K = rbf_kernel(X, X, gamma)
    Q = np.outer(y, y) * K
    
    start_time = time.time()
    func_eval = []
    
    final_gap = 0
    iteration = 0
    
    for it in range(max_iter):
        iteration = it
        
        # 1. Compute Gradient: Q*alpha - 1
        grad = np.dot(Q, alpha) - 1.0
        
        # 2. Select Working Set
        ws_idx, gap = get_working_set_indices(alpha, grad, y, C, q)
        final_gap = gap
        
        if gap < tol or len(ws_idx) < 2:
           break
        
        # 3. Setup Subproblem
        current_alpha_sub = alpha[ws_idx]
        Q_sub = Q[np.ix_(ws_idx, ws_idx)]
        y_sub = y[ws_idx]
        linear_term = grad[ws_idx] - np.dot(Q_sub, current_alpha_sub) 
        
        # 4. Solve
        alpha_new_sub, fun_count = solve_subproblem(Q_sub, y_sub, linear_term, C, current_alpha_sub)
        
        func_eval.append(fun_count)

        # 5. Update
        alpha[ws_idx] = alpha_new_sub

    end_time = time.time()
    
    # Final Objective
    final_obj = 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
    
    # Calculate Bias b
    sv_indices = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
    if len(sv_indices) > 0:
        b_list = [y[k] - np.dot(alpha * y, K[:, k]) for k in sv_indices]
        b = np.mean(b_list)
    else:
        b = 0.0

    stats = {
        'time': end_time - start_time,
        'iterations': iteration,
        'final_obj': final_obj,
        'gap': final_gap,
        'func_eval': sum(func_eval),
        'status': 'Converged' if iteration < max_iter else 'Max Iter'
    }
    
    return alpha, b, stats