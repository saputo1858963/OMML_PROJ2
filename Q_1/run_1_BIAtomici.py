
# -*- coding: utf-8 -*-
"""
RUN Question 1 - Gruppo: BIAtomici

Questo è l'UNICO file che verrà eseguito in fase di verifica.
DEVE stampare SOLO:
- Setting values of the hyperparameters
- Classification rate on the training set (% instances correctly classified)
- Classification rate on the test set (% instances correctly classified)
- The confusion matrix
- Time necessary for the optimization
- Number of optimization iterations
- Difference between m(alpha) and M(alpha)
- The final value of the Dual SVM objective function f(alpha*)
- Solver status

Rispettare rigorosamente il formato (nessun logging di progresso).
"""

import numpy as np
from functions_1_BIAtomici import (
    extract_data, scale_minmax_01, grid_search_cv, train_svm_dual_frank_wolfe,
    predict, accuracy_percent, confusion_matrix_binary
)

def main():
    # 1) Carica dati
    Xtr, ytr, Xte, yte = extract_data()

    # 2) Scaling (richiesto dalla traccia)
    Xtr = scale_minmax_01(Xtr)
    Xte = scale_minmax_01(Xte)

    # 3) Scelta kernel e griglia iperparametri
    kernel = 'rbf'            # puoi cambiare in 'poly' se desideri
    grid_C = [0.1, 1.0, 10.0, 100.0]
    # gamma per RBF; se usi 'poly' intendi gamma come grado del polinomio
    grid_gamma = [1e-3, 1e-2, 1e-1, 1.0]

    # 4) k-fold CV per trovare (C, gamma)
    cv = grid_search_cv(
        Xtr, ytr, kernel=kernel,
        grid_C=grid_C, grid_gamma=grid_gamma,
        k_fold=5, max_iter=300, tol_kkt=1e-3, seed=1
    )
    C_best = cv['best_params']['C']
    gamma_best = cv['best_params']['gamma']

    # 5) Allena modello finale su TUTTO il training set con (C*, gamma*)
    model = train_svm_dual_frank_wolfe(
        Xtr, ytr, C=C_best, kernel=kernel, gamma=gamma_best,
        max_iter=800, tol_kkt=5e-4, seed=1, verbose=False
    )

    # 6) Predizioni e metriche
    yhat_tr = predict(model, Xtr)
    yhat_te = predict(model, Xte)

    acc_tr = accuracy_percent(ytr, yhat_tr)
    acc_te = accuracy_percent(yte, yhat_te)
    cm = confusion_matrix_binary(yte, yhat_te)

    # 7) STAMPE (SOLO quanto richiesto)
    #    Nota: classi: +1 = pullover (label 2), -1 = dress (label 3)
    print(f"Kernel: {model['kernel']}, C: {model['C']}, gamma: {model['gamma']}")
    print(f"Training accuracy (%): {acc_tr:.2f}")
    print(f"Test accuracy (%): {acc_te:.2f}")
    print("Confusion matrix (rows=true [-1,+1], cols=pred [-1,+1]):")
    print(cm.tolist())  # list per una stampa pulita senza array prefix
    print(f"Optimization time (s): {model['train_time_sec']:.6f}")
    print(f"Number of iterations: {model['iterations']}")
    print(f"m(alpha) - M(alpha): {model['kkt_m_minus_M']:.6e}")
    print(f"Dual objective f(alpha*): {model['final_obj']:.10e}")
    print(f"Solver status: {model['status']}")

if __name__ == "__main__":
    main()
