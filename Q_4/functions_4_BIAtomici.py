from typing import Any, Dict, List
import numpy as np
import pandas as pd
import os
import time

def extract_data_Q4():
    script_dir = os.path.dirname(__file__)  # Script folder
    base_dir = os.path.dirname(script_dir)  # folder of the folder
    file_train = os.path.join(base_dir, "FashionMNIST", "fashion-mnist_train.csv")
    file_test = os.path.join(base_dir, "FashionMNIST", "fashion-mnist_test.csv")

    train = pd.read_csv(file_train)
    test = pd.read_csv(file_test)

    labels_of_interest = [2, 3, 6]
    
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    n_train_per_class = 1000
    n_test_per_class = 200

    for label in labels_of_interest:
        # Training extraction
        df_class_train = train[train['label'] == label]
        X_curr = df_class_train.values[:, 1:][:n_train_per_class]
        Y_curr = df_class_train.values[:, 0][:n_train_per_class]
        X_train_list.append(X_curr)
        Y_train_list.append(Y_curr)

        # Test extraction
        df_class_test = test[test['label'] == label]
        X_curr_test = df_class_test.values[:, 1:][:n_test_per_class]
        Y_curr_test = df_class_test.values[:, 0][:n_test_per_class]
        X_test_list.append(X_curr_test)
        Y_test_list.append(Y_curr_test)

    X_train = np.concatenate(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    X_test = np.concatenate(X_test_list)
    Y_test = np.concatenate(Y_test_list)

    # Data scaling
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Shuffle 
    np.random.seed(1)
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    return X_train, Y_train, X_test, Y_test

def rbf_kernel_matrix(X1, X2, gamma):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

# SVM for binary classification {-1, +1}
class BinarySVM:
    def __init__(self, C=1.0, gamma=0.01, tol=1e-3, max_iter=1000):  #  constructor
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.training_stats = {}

    #  Implementation of a decomposition method (SMO-MVP)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        alpha = np.zeros(n_samples)
        self.b = 0
        
        # Precalcola kernel matrix
        K = rbf_kernel_matrix(X, X, self.gamma)
        
        start_time = time.time()
        iter_count = 0
        
        # alpha * y * K
        decision = np.zeros(n_samples) 
        
        while iter_count < self.max_iter:

            #grad_f = y * decision - 1.0
            # MVP indices
            score = y - decision  # this is actually (-gradient of f * y), we've just substitute Q_{ij} = y_i y_j K(x_i, x_j)
            
            R = ((alpha < self.C - 1e-7) & (y == 1)) | ((alpha > 1e-7) & (y == -1))
            S = ((alpha < self.C - 1e-7) & (y == -1)) | ((alpha > 1e-7) & (y == 1))
            
            if not np.any(R) or not np.any(S):
                break # no candidates
                
            # Filter non valid scores
            score_R = np.where(R, score, -np.inf) # if index i is in R we take its score, else we set its score to -infinite
            score_S = np.where(S, score, np.inf)
            
            i = np.argmax(score_R)
            j = np.argmin(score_S)
            
            m_alpha = score[i] 
            M_alpha = score[j] 
            
            # Optimality control
            if (m_alpha - M_alpha) < self.tol:
                break
                
            # Save old values
            alpha_i_old = alpha[i]
            alpha_j_old = alpha[j]
            
            # Compute bounds L e H
            if y[i] != y[j]:
                L = max(0, alpha[j] - alpha[i])
                H = min(self.C, self.C + alpha[j] - alpha[i])
            else:
                L = max(0, alpha[i] + alpha[j] - self.C)
                H = min(self.C, alpha[i] + alpha[j])
                
            if L == H:
                continue
            
            # Second derivative of the dual objective function
            eta = 2 * K[i, j] - K[i, i] - K[j, j]
            
            if eta >= 0: # eta shouldn't be negative
                iter_count += 1
                continue
            
            # Update alpha_j
            E_i = decision[i] - y[i]
            E_j = decision[j] - y[j]
            
            alpha[j] -= y[j] * (E_i - E_j) / eta
            alpha[j] = np.clip(alpha[j], L, H)
            
            if abs(alpha[j] - alpha_j_old) < 1e-5:
                iter_count += 1
                continue
                
            # Update Alpha i (for the condition alpha*y = 0, if alpha_j decreases, alpha_i must increase or viceversa)
            alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
            
            # Efficient update through 'decision'
            delta_i = alpha[i] - alpha_i_old
            delta_j = alpha[j] - alpha_j_old
            decision += delta_i * y[i] * K[:, i] + delta_j * y[j] * K[:, j]
            
            iter_count += 1

        # Compute the Bias with SV
        sv = (alpha > 1e-5) & (alpha < self.C - 1e-5)
        if np.any(sv):
            self.b = np.mean(y[sv] - decision[sv])
        else:
            self.b = 0.0

        self.alpha = alpha
        sv_indices = alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha_sv = alpha[sv_indices]
        
        # Compute dual objective funztion 
        term1 = 0.5 * np.sum(alpha * y * decision)
        term2 = np.sum(alpha)
        dual_obj = term1 - term2
        
        if iter_count < self.max_iter:
            status = "optimal"
        else:
            status = "max_iter_reached"

        self.training_stats = {
            'time': time.time() - start_time,
            'iterations': iter_count,
            'dual_obj': dual_obj,
            'kkt_viol': (m_alpha - M_alpha) if iter_count > 0 else 0, # Stima finale violazione
            'n_cols': n_samples,
            'status': status
        }

    def predict(self, X):
        K_new = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        decision = (self.alpha_sv * self.support_vector_labels) @ K_new.T + self.b
        return np.sign(decision)

# One_Against_One implementation
class MulticlassSVM_OAO:
    def __init__(self, C=1.0, gamma=0.01):
        self.C = C
        self.gamma = gamma
        self.classifiers = []
        self.classes = []
        self.stats = {
            'time': 0,
            'iterations': 0,
            'kkt_viol': 0,
            'dual_obj': 0,
            'n_cols': 0,
            'status': 'optimal'
        }

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        start_global = time.time()
        
        # Find all the possible couples
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                c1 = self.classes[i]
                c2 = self.classes[j]
                
                # Estrai dati per le due classi
                mask = (y == c1) | (y == c2)
                X_pair = X[mask]  # take the images that corresponds to the labels of the classes
                y_pair = y[mask]  # take the labels that corresponds to c1 and c2
                
                # Convert labels to +1 (c1) e -1 (c2)
                y_binary = np.where(y_pair == c1, 1.0, -1.0)
                
                # Train Binary SVM, one for each specific couple of classes
                model = BinarySVM(C=self.C, gamma=self.gamma, max_iter=500)
                model.fit(X_pair, y_binary)
                
                if model.training_stats['status'] != 'optimal':
                   self.stats['status'] = 'suboptimal/max_iter'

                self.classifiers.append({
                    'c1': c1,
                    'c2': c2,
                    'model': model
                })
                
                self.stats['iterations'] += model.training_stats['iterations']
                self.stats['n_cols'] += model.training_stats['n_cols']
                self.stats['kkt_viol'] = max(self.stats['kkt_viol'], model.training_stats['kkt_viol'])
                self.stats['dual_obj'] += model.training_stats['dual_obj']

        self.stats['time'] = time.time() - start_global

    # The voting procedure
    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        for clf_info in self.classifiers:
            model = clf_info['model']
            c1 = clf_info['c1']
            c2 = clf_info['c2']
            
            pred = model.predict(X)
            
            # pred == 1 -> c1, pred == -1 -> c2
            idx_c1 = class_to_idx[c1]
            idx_c2 = class_to_idx[c2]
            
            for k in range(n_samples):
                if pred[k] > 0:
                    votes[k, idx_c1] += 1  # vote for class c1
                else:
                    votes[k, idx_c2] += 1  # vote for class c2
                    
        # Argmax of votes
        predicted_indices = np.argmax(votes, axis=1)  # final verdict
        return self.classes[predicted_indices]

# Confusion matrix
def compute_confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            i = label_to_idx[yt]
            j = label_to_idx[yp]
            cm[i, j] += 1
    return cm

def print_confusion_matrix_formatted(cm, labels):
    print("Confusion Matrix:")
    print("      Pred", end="")
    for l in labels: print(f"{l:>5}", end="")
    print("\nTrue")
    for i, l in enumerate(labels):
        print(f"{l:<5}", end="")
        for val in cm[i]:
            print(f"{val:5d}", end="")
        print()

# ============================================================
# K-fold Cross-Validation & Grid Search
# ============================================================

def accuracy_percent(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100.0

def kfold_indices(n: int, k: int, seed: int = 1, shuffle: bool = True) -> List[np.ndarray]:
    rng = np.random.default_rng(seed) 
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def grid_search_cv(
    X: np.ndarray,
    y: np.ndarray,
    grid_C: List[float],
    grid_gamma: List[float],
    k_fold: int = 5,
    max_iter: int = 300,
    tol_kkt: float = 1e-3,
    seed: int = 1,
) -> Dict[str, Any]:
    
    n = X.shape[0]
    folds = kfold_indices(n, k_fold, seed=seed, shuffle=True)

    results = [] 
    best_score = -np.inf
    best_params = None

    for C in grid_C:
        for gamma in grid_gamma:
            acc_list = [] 
            for k in range(k_fold):
                val_idx = folds[k]
                tr_idx = np.hstack([folds[j] for j in range(k_fold) if j != k])
                
                Xtr, ytr = X[tr_idx], y[tr_idx]
                Xval, yval = X[val_idx], y[val_idx]
                
                # Train the model using BinarySVM class
                model = BinarySVM(C=C, gamma=gamma, tol=tol_kkt, max_iter=max_iter)
                model.fit(Xtr, ytr) # Fit calcola alpha e b

                yhat_val = model.predict(Xval) # prediction
                acc = accuracy_percent(yval, yhat_val)
                acc_list.append(acc)

            mean_acc = float(np.mean(acc_list))
            
            results.append({'C': C, 'gamma': gamma, 'val_acc': mean_acc})
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = {'C': C, 'gamma': gamma}

    return {
        'best_params': best_params,
        'best_score': best_score,
        'results': results
    }