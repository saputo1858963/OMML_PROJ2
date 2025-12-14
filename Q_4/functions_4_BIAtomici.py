import numpy as np
import pandas as pd
import os
import time

def extract_data_Q4():
    """
    Extracts Fashion MNIST data for Question 4 (3 classes: Pullover, Dress, Shirt).
    Classes:
    - 2: Pullover
    - 3: Dress
    - 6: Shirt
    
    Returns:
        X_train, Y_train, X_test, Y_test: Scaled data and labels.
    """
    # Adjust path to step out of Q_4 folder and find FashionMNIST
    train_path = os.path.join('..', 'FashionMNIST', 'fashion-mnist_train.csv')
    test_path = os.path.join('..', 'FashionMNIST', 'fashion-mnist_test.csv')

    if not os.path.exists(train_path):
        # Fallback if running from root
        train_path = os.path.join('FashionMNIST', 'fashion-mnist_train.csv')
        test_path = os.path.join('FashionMNIST', 'fashion-mnist_test.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Q4 requires classes 2, 3, and 6
    indexes = [2, 3, 6]

    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:, 1:].astype(np.float64)
    Y = train_set.values[:, 0]

    X_Test_All = test_set.values[:, 1:].astype(np.float64)
    Y_Test_All = test_set.values[:, 0]

    # Sampling 1000 per class for training as per pattern in extractor.py
    X_train_list = []
    Y_train_list = []
    
    # Sampling 200 per class for testing
    X_test_list = []
    Y_test_list = []

    for lbl in indexes:
        # Train indices
        ind = np.where(Y == lbl)[0]
        # Test indices
        ind_test = np.where(Y_Test_All == lbl)[0]

        X_train_list.append(X[ind[:1000]])
        Y_train_list.append(np.full(1000, lbl))
        
        X_test_list.append(X_Test_All[ind_test[:200]])
        Y_test_list.append(np.full(200, lbl))

    X_train = np.concatenate(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    X_test = np.concatenate(X_test_list)
    Y_test = np.concatenate(Y_test_list)

    # Shuffle
    np.random.seed(1)
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]
    
    # MinMax Scaling to [0, 1] (Crucial for SVM convergence)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

# --- Kernels ---
def rbf_kernel_matrix(X1, X2, gamma):
    # Vectorized RBF Kernel
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

# --- Simplified SMO/MVP Solver for Binary SVM ---
class BinarySVM_MVP:
    def __init__(self, C, gamma, tol=1e-3, max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.obj_history = []
        self.final_obj = 0
        self.kkt_viol = 0
        self.iterations = 0

    def fit(self, X, y):
        # y must be +1/-1
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Precompute Kernel Matrix (might be heavy for large N, but ok for N=3000)
        K = rbf_kernel_matrix(X, X, self.gamma)
        
        # MVP (Most Violating Pair) Loop
        # Gradient of Dual: G_i = y_i * (sum alpha_j y_j K_ij + b) - 1 ... wait
        # Standard Dual Gradient G_i = (sum alpha_j y_j K_ij) - 1
        # Optimality conditions are on y_i * gradient.
        
        start_time = time.time()
        
        for it in range(self.max_iter):
            self.iterations = it
            
            # Decision function (excluding b first for gradient comp)
            # f(x) = sum alpha_j y_j K(x_j, x)
            # Q_ij = y_i y_j K_ij
            # grad_i = sum_j (Q_ij alpha_j) - 1
            
            # Efficient update of gradient? For simplicity, full compute or partial
            decision = (self.alpha * y) @ K # shape (n,)
            grad = decision - 1.0
            
            # KKT Conditions check
            # y_i * grad_i = y_i * f(x_i) - y_i = y_i * f(x_i) - y_i ...
            # Let's use standard MVP selection criteria
            # r_i = y_i * grad_i  (This is roughly checking y*f(x) - 1 relation)
            
            # I_up: alpha < C if y=1, alpha > 0 if y=-1
            # I_down: alpha > 0 if y=1, alpha < C if y=-1
            
            # Define projection of gradient: -y_i * grad_i
            # We want to Maximize Dual.
            # Violated i (up) maximizes -y * grad
            # Violated j (down) minimizes -y * grad
            
            yd = y * decision # This is y_i * (sum alpha y K)
            
            # Optimality:
            # if alpha = 0 => yd >= 1 (approx)
            # if 0 < alpha < C => yd = 1
            # if alpha = C => yd <= 1
            
            # Bias b is implicitly handled in full SMO, but for decomposition usually derived.
            # For MVP selection we use the gradient of the objective function.
            
            # Let's use the violation metric:
            # G = y * f(x) (where f doesn't include b yet)
            # Violated if:
            # 1. alpha < C and G < 1 (should increase alpha) -> Candidate for I_up (if we view it as lagrangian)
            
            # Keerthi MVP Selection:
            # i = argmax_{k in I_up}  {-y_k * grad_k}
            # j = argmin_{k in I_down} {-y_k * grad_k}
            # where grad_k is gradient of Dual Objective to be MINIMIZED? 
            # Usual Dual is MAXimized. Let's assume Min 1/2 aQa - 1.a
            # Grad = Q alpha - 1.
            # F_i = - y_i * Grad_i = y_i - (sum alpha_j y_j K_ij) * y_i
            
            F = y - decision # F_i = y_i - f_i (without b)
            
            mask_up = ((y == 1) & (self.alpha < self.C)) | ((y == -1) & (self.alpha > 0))
            mask_down = ((y == 1) & (self.alpha > 0)) | ((y == -1) & (self.alpha < self.C))
            
            # If no valid indices, optimum reached
            if not np.any(mask_up) or not np.any(mask_down):
                break
                
            i = -1
            j = -1
            
            # Use masked arrays to find min/max
            F_up = F.copy()
            F_up[~mask_up] = -np.inf
            i = np.argmax(F_up)
            
            F_down = F.copy()
            F_down[~mask_down] = np.inf
            j = np.argmin(F_down)
            
            # Check optimality gap
            diff = F[i] - F[j]
            self.kkt_viol = diff
            
            if diff < self.tol:
                break
                
            # Update Step for pair (i, j)
            # Analytic solution for 2 variables
            # maximize W(a_i, a_j) s.t. constraint
            
            # s = y_i * y_j
            old_ai = self.alpha[i]
            old_aj = self.alpha[j]
            
            eta = K[i,i] + K[j,j] - 2*K[i,j]
            
            if eta <= 0:
                eta = 1e-12 # Numeric stability
                
            delta_alpha_j = y[j] * (F[i] - F[j]) / eta
            
            new_aj = old_aj + delta_alpha_j
            
            # Clip new_aj
            sum_s = old_ai * y[i] + old_aj * y[j] # Const: a_i y_i + a_j y_j = k
            
            if y[i] != y[j]:
                L = max(0, old_aj - old_ai)
                H = min(self.C, self.C + old_aj - old_ai)
            else:
                L = max(0, old_ai + old_aj - self.C)
                H = min(self.C, old_ai + old_aj)
                
            if new_aj > H: new_aj = H
            if new_aj < L: new_aj = L
            
            new_ai = old_ai + y[i] * y[j] * (old_aj - new_aj)
            
            self.alpha[i] = new_ai
            self.alpha[j] = new_aj
            
        # Compute b
        # Average over support vectors 0 < alpha < C
        sv_ind = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        if np.any(sv_ind):
             self.b = np.mean(y[sv_ind] - decision[sv_ind])
        else:
             # Fallback
             self.b = 0
             
        # Store Model
        mask_sv = self.alpha > 1e-5
        self.support_vectors = X[mask_sv]
        self.support_vector_labels = y[mask_sv]
        self.support_vector_alphas = self.alpha[mask_sv]
        
        # Dual Obj Value: Sum alpha - 0.5 alpha Q alpha
        # = Sum alpha - 0.5 * alpha * (decision) ... wait decision was Q alpha? yes
        # Recalculate decision with final alpha
        final_decision = (self.alpha * y) @ K
        self.final_obj = np.sum(self.alpha) - 0.5 * np.dot(self.alpha * y, final_decision)

    def decision_function(self, X):
        K_new = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        # f(x) = sum alpha_i y_i K(x_i, x) + b
        return (K_new @ (self.support_vector_alphas * self.support_vector_labels)) + self.b
        
    def predict(self, X):
        return np.sign(self.decision_function(X))

# --- Multiclass OAA Wrapper ---
class MulticlassSVM_OAA:
    def __init__(self, C=1.0, gamma=0.1, tol=1e-2, max_iter=500):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.models = [] # List of BinarySVM_MVP
        self.classes = []
        self.total_obj = 0
        self.total_iter = 0
        self.avg_kkt = 0
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []
        
        objs = []
        iters = []
        kkts = []
        
        # Train One vs All
        for c in self.classes:
            # Create binary labels: 1 for current class, -1 for others
            y_binary = np.where(y == c, 1.0, -1.0)
            
            svm = BinarySVM_MVP(self.C, self.gamma, self.tol, self.max_iter)
            svm.fit(X, y_binary)
            
            self.models.append(svm)
            objs.append(svm.final_obj)
            iters.append(svm.iterations)
            kkts.append(svm.kkt_viol)
            
        self.total_obj = np.sum(objs)
        self.total_iter = np.sum(iters)
        self.avg_kkt = np.mean(kkts)
        
    def predict(self, X):
        # Compute scores for all classes
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, model in enumerate(self.models):
            scores[:, idx] = model.decision_function(X)
        
        # Argmax to get class index, then map to label
        max_indices = np.argmax(scores, axis=1)
        return self.classes[max_indices]

def get_confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for yt, yp in zip(y_true, y_pred):
        if yt in class_to_idx and yp in class_to_idx:
            cm[class_to_idx[yt], class_to_idx[yp]] += 1
    return cm