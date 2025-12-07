"""
Question 1 - SVM Dual con Frank-Wolfe (Conditional Gradient Method)
Gruppo: BIAtomici

Questo modulo contiene:
- extract_data(): estrae i dati dalle CSV FashionMNIST (classi 2 vs 3) e
  crea un train/test binario con etichette +1 (label 2) e -1 (label 3).
- Scaling delle feature.
- Kernel RBF e (opzionalmente) polynomial.
- Addestramento SVM duale via Frank-Wolfe (minimizzazione convessa del duale).
- Calcolo di b, funzione decisionale, accuratezza, confusion matrix.
- k-fold cross-validation con grid search su (C, gamma).

Restrizioni: niente librerie ML auto-allenanti (sklearn, torch, ecc.).
Solo numpy/pandas/standard lib.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


# ============================================================
# 1) Estrazione dati (come da traccia)
# ============================================================
def extract_data():
    """
    Load and prepare FashionMNIST folder containing the training and testing data (csv files).
    Make sure the folder 'FashionMNIST' is inside your project directory (i.e. at the same level of your Q_i folders).
    """

    train = pd.read_csv(r"C:\Users\gluca\Desktop\OMML_PROJ2\FashionMNIST\fashion-mnist_train.csv")
    test = pd.read_csv(r"C:\Users\gluca\Desktop\OMML_PROJ2\FashionMNIST\fashion-mnist_test.csv")

    # We need to extract only the items with label 2 and 3:
    indexes = [2, 3]

    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:, 1:]
    Y = train_set.values[:, 0]

    X_Test = test_set.values[:, 1:]
    Y_Test = test_set.values[:, 0]

    ind_2 = np.where(Y == 2)
    ind_2_Test = np.where(Y_Test == 2)

    ind_3 = np.where(Y == 3)
    ind_3_Test = np.where(Y_Test == 3)

    """
    Only a subset of 1000 samples per class will be used.
    To train a SVM in case of binary classification we have to convert the labels of the two classes of interest into '+1' and '-1'.
    """

    X2 = X[ind_2[0][:1000]]
    Y2 = np.ones(1000)

    X3 = X[ind_3[0][:1000]]
    Y3 = -np.ones(1000)

    X2test = X_Test[ind_2_Test[0][:200]]
    Y2test = np.ones(X2test.shape[0])

    X3test = X_Test[ind_3_Test[0][:200]]
    Y3test = -np.ones(X3test.shape[0])

    X_train = np.concatenate((X2, X3))  # X_train (2000x784): pixel delle immagini train (scala grezza 0–255)
    Y_train = np.concatenate((Y2, Y3))  # Y_train (2000,): etichette {+1,-1}

    X_test = np.concatenate((X2test, X3test))  # X_test (400×784)
    Y_test = np.concatenate((Y2test, Y3test))  # Y_test (400,)

    # Reshuffle training data
    np.random.seed(1)
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    return X_train, Y_train, X_test, Y_test


# ============================================================
# 2) Preprocessing & Utilità
# ============================================================

# Normalizzare i pixel in [0,1]
def scale_minmax_01(X: np.ndarray) -> np.ndarray:
    """Scala le feature in [0,1] (pixel/255)."""
    return X.astype(np.float64) / 255.0  # se X è uint8 lo converte in folat64


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix 2x2 per classi {-1, +1}.
    Le righe corrispondono alle classi vere [riga 0: -1, riga 1: +1], Colonne sono le classi predette = predetti nell'ordine [colonna 0: -1, colonna 1:  +1].
    """
    cm = np.zeros((2, 2), dtype=int)  # inizializzazione a 0
    # Mappatura delle classi da {-1,+1} a {0,1} tramite astype() --> usefuk for the loop below
    t = (y_true > 0).astype(int)  # y_true: etichette vere
    p = (y_pred > 0).astype(int)  # y_pred: etichette predette
    for i in range(2):
        for j in range(2):
            cm[i, j] = np.sum((t == i) & (p == j))
    return cm


def accuracy_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 100.0 * np.mean(y_true == y_pred)


# ============================================================
# 3) Kernel
# ============================================================
def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Distanze quadrate euclidee tutte-le-coppie in modo vettoriale."""
    X2 = np.sum(X * X, axis=1)[:, None]
    Y2 = np.sum(Y * Y, axis=1)[None, :]
    return np.maximum(X2 + Y2 - 2.0 * X @ Y.T, 0.0)


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """K(x,y) = exp(-gamma * ||x - y||^2)"""
    D2 = _sq_dists(X, Y)
    return np.exp(-gamma * D2)


def poly_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """K(x,y) = (x^T y + 1)^gamma (qui gamma è il grado, tipicamente intero >= 1)."""
    return (X @ Y.T + 1.0) ** gamma


def compute_kernel(XA: np.ndarray, XB: np.ndarray, kernel: str, gamma: float) -> np.ndarray:
    if kernel.lower() == 'rbf':
        return rbf_kernel(XA, XB, gamma)
    elif kernel.lower() == 'poly':
        return poly_kernel(XA, XB, gamma)
    else:
        raise ValueError("Kernel non supportato: scegli 'rbf' o 'poly'.")


# ============================================================
# 4) SVM Dual via Frank-Wolfe
# ============================================================
def _dual_objective(alpha: np.ndarray, Q: np.ndarray) -> float:
    """
    f(alpha) = 0.5 * alpha^T Q alpha - 1^T alpha
    (problema di minimizzazione convessa equivalente al duale SVM)
    """
    return 0.5 * alpha @ (Q @ alpha) - np.sum(alpha)


def _dual_gradient(alpha: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """g = grad f = Q alpha - 1"""
    return Q @ alpha - np.ones_like(alpha)


def _linear_minimization_oracle_FW(g: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """
    Oracle di minimizzazione lineare per l'insieme
    P = { alpha in [0,C]^n : y^T alpha = 0 }.
    Un vertice di P è ottenibile selezionando una coppia (p,n) con y_p=+1, y_n=-1
    e ponendo alpha_p = C, alpha_n = C, gli altri 0. Scegliamo gli argmin del gradiente
    per ciascun gruppo.
    """
    # Indici per classi
    P_idx = np.where(y > 0)[0]
    N_idx = np.where(y < 0)[0]
    # argmin del gradiente in ciascun gruppo
    ip = P_idx[np.argmin(g[P_idx])]
    ineg = N_idx[np.argmin(g[N_idx])]
    s = np.zeros_like(g)
    s[ip] = C
    s[ineg] = C
    return s


def _compute_b_from_K(K: np.ndarray, y: np.ndarray, alpha: np.ndarray, C: float, tol: float = 1e-8) -> float:
    """
    Calcola b usando i vettori di supporto con 0 < alpha_i < C.
    Se non ce ne sono, usa tutti gli alpha > tol come fallback.
    """
    ay = alpha * y
    decision_no_b = K @ ay  # f_i senza b
    margin = np.where((alpha > tol) & (alpha < C - tol))[0]
    if margin.size > 0:
        b_i = y[margin] - decision_no_b[margin]
        return float(np.mean(b_i))
    # fallback
    sv = np.where(alpha > tol)[0]
    if sv.size > 0:
        b_i = y[sv] - decision_no_b[sv]
        return float(np.mean(b_i))
    # casi degeneri: nessun SV -> b=0
    return 0.0


def _kkt_m_M(K: np.ndarray, y: np.ndarray, alpha: np.ndarray, C: float, eps: float = 1e-8) -> Tuple[float, float, float]:
    """
    Calcola m(alpha), M(alpha) e la loro differenza (criterio KKT stile SMO):
    G_i = y_i * (K @ (alpha*y))_i - 1
    I_up  = {i | (y_i=+1 and alpha_i < C) or (y_i=-1 and alpha_i > 0)}
    I_low = {i | (y_i=+1 and alpha_i > 0) or (y_i=-1 and alpha_i < C)}
    m = max_{i in I_up}  (-G_i)
    M = min_{i in I_low} (-G_i)
    """
    ay = alpha * y
    f_no_b = K @ ay
    G = y * f_no_b - 1.0

    I_up = np.where(((y > 0) & (alpha < C - eps)) | ((y < 0) & (alpha > eps)))[0]
    I_low = np.where(((y > 0) & (alpha > eps)) | ((y < 0) & (alpha < C - eps)))[0]

    m = np.max(-G[I_up]) if I_up.size > 0 else np.inf
    M = np.min(-G[I_low]) if I_low.size > 0 else -np.inf
    gap = m - M
    return float(m), float(M), float(gap)


def train_svm_dual_frank_wolfe(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    kernel: str = 'rbf',
    gamma: float = 1e-2,
    max_iter: int = 500,
    tol_kkt: float = 1e-3,
    seed: int = 1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Allena la SVM duale minimizzando f(alpha) con Frank-Wolfe.
    Ritorna un dizionario modello con: alpha, b, support_idx, info ottimizzazione, ecc.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    y = y.astype(np.float64).copy()

    # Kernel train K e matrice Q = (y y^T) ∘ K
    K = compute_kernel(X, X, kernel=kernel, gamma=gamma)
    Q = (y[:, None] * y[None, :]) * K

    # Inizializzazione: alpha=0 (ammissibile: 0<=alpha<=C e y^T alpha = 0)
    alpha = np.zeros(n, dtype=np.float64)

    history_obj = []
    status = 'running'
    t0 = time.time()
    it = 0

    for it in range(1, max_iter + 1):
        g = _dual_gradient(alpha, Q)
        s = _linear_minimization_oracle_FW(g, y, C)
        d = s - alpha

        # Line-search esatta per quadratica: gamma* = clip( -g^T d / (d^T Q d), [0,1] )
        denom = d @ (Q @ d)
        numer = g @ d
        if denom > 0:
            step = np.clip(-numer / denom, 0.0, 1.0)
        else:
            # direzione lineare; se numer < 0, prendi passo pieno
            step = 1.0 if numer < 0 else 0.0

        alpha = alpha + step * d
        # (per costruzione, alpha resta in [0,C] e soddisfa y^T alpha = 0)

        fval = _dual_objective(alpha, Q)
        history_obj.append(fval)

        # Criterio di arresto KKT m-M
        m, M, gap = _kkt_m_M(K, y, alpha, C)
        if verbose:
            print(f"[FW] it={it:4d}  f={fval:.6e}  step={step:.3e}  m-M={gap:.3e}")
        if gap <= tol_kkt:
            status = 'optimal'
            break

    cpu_time = time.time() - t0

    # Calcolo b
    b = _compute_b_from_K(K, y, alpha, C)

    # Support vectors (alpha > 0)
    sv_idx = np.where(alpha > 1e-12)[0]

    model = {
        'alpha': alpha,
        'b': b,
        'support_idx': sv_idx,
        'kernel': kernel,
        'gamma': gamma,
        'C': C,
        'train_time_sec': cpu_time,
        'iterations': it,
        'final_obj': history_obj[-1] if len(history_obj) else None,
        'status': status,
        'kkt_m_minus_M': gap if 'gap' in locals() else None,
        'K_train': K,  # utile per valutazioni su train veloci
        'X_train': X,
        'y_train': y,
    }
    return model


# ============================================================
# 5) Predizione
# ============================================================
def decision_function(model: Dict[str, Any], Xq: np.ndarray) -> np.ndarray:
    """f(x) = sum_j alpha_j y_j K(x_j, x) + b"""
    Xtr = model['X_train']
    ytr = model['y_train']
    alpha = model['alpha']
    b = model['b']
    Kqx = compute_kernel(Xq, Xtr, kernel=model['kernel'], gamma=model['gamma'])
    return Kqx @ (alpha * ytr) + b


def predict(model: Dict[str, Any], Xq: np.ndarray) -> np.ndarray:
    return np.sign(decision_function(model, Xq)).astype(np.float64)


# ============================================================
# 6) K-fold Cross-Validation & Grid Search
# ============================================================
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
    kernel: str,
    grid_C: List[float],
    grid_gamma: List[float],
    k_fold: int = 5,
    max_iter: int = 300,
    tol_kkt: float = 1e-3,
    seed: int = 1,
) -> Dict[str, Any]:
    """
    Esegue k-fold CV su griglia (C, gamma) per il kernel scelto.
    Restituisce best params e tabella risultati.
    """
    n = X.shape[0]
    folds = kfold_indices(n, k_fold, seed=seed, shuffle=True)

    results = []
    best_score = -np.inf
    best_params = None

    # Pre-compute niente (si ricalcola per ogni training fold per semplicità e robustezza)
    for C in grid_C:
        for gamma in grid_gamma:
            acc_list = []
            for k in range(k_fold):
                val_idx = folds[k]
                tr_idx = np.hstack([folds[j] for j in range(k_fold) if j != k])
                Xtr, ytr = X[tr_idx], y[tr_idx]
                Xval, yval = X[val_idx], y[val_idx]

                model = train_svm_dual_frank_wolfe(
                    Xtr, ytr, C=C, kernel=kernel, gamma=gamma,
                    max_iter=max_iter, tol_kkt=tol_kkt, seed=seed, verbose=False
                )
                yhat_val = predict(model, Xval)
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
