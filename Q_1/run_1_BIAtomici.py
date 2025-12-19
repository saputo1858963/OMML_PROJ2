from functions_1_BIAtomici import (
    extract_data, scale_minmax_01, grid_search_cv, train_svm_dual_frank_wolfe,
    predict, accuracy_percent, confusion_matrix_binary
)

def main():
    # 1) Load data
    Xtr, ytr, Xte, yte = extract_data()

    # 2) Scaling 
    Xtr = scale_minmax_01(Xtr)
    Xte = scale_minmax_01(Xte)

    # 3) Kernel and grid hyperparameter selection
    kernel = 'rbf'            
    grid_C = [0.1, 1.0, 10.0, 100.0]
    grid_gamma = [1e-3, 1e-2, 1e-1, 1.0]

    # 4) k-fold CV to find (C, gamma)
    cv = grid_search_cv(
        Xtr, ytr, kernel=kernel,
        grid_C=grid_C, grid_gamma=grid_gamma,
        k_fold=5, max_iter=300, tol_kkt=1e-3, seed=1
    )
    C_best = cv['best_params']['C']
    gamma_best = cv['best_params']['gamma']

    # 5) Train final model on all training set with (C*, gamma*)
    model = train_svm_dual_frank_wolfe(
        Xtr, ytr, C=C_best, kernel=kernel, gamma=gamma_best,
        max_iter=500, tol_kkt=5e-4, seed=1, verbose=False
    )

    # 6) Predictions and metrics
    yhat_tr = predict(model, Xtr)
    yhat_te = predict(model, Xte)

    acc_tr = accuracy_percent(ytr, yhat_tr)
    acc_te = accuracy_percent(yte, yhat_te)
    cm = confusion_matrix_binary(yte, yhat_te)

    #    Note: classes: +1 = pullover (label 2), -1 = dress (label 3)
    print(f"Kernel: {model['kernel']}, C: {model['C']}, gamma: {model['gamma']}")
    print(f"Training accuracy (%): {acc_tr:.2f}")
    print(f"Test accuracy (%): {acc_te:.2f}")
    print("Confusion matrix (rows=true [-1,+1], cols=pred [-1,+1]):")
    print(cm.tolist())  
    print(f"Optimization time (s): {model['train_time_sec']:.6f}")
    print(f"Number of iterations: {model['iterations']}")
    print(f"m(alpha) - M(alpha): {model['kkt_m_minus_M']:.6e}")
    print(f"Dual objective f(alpha*): {model['final_obj']:.10e}")
    print(f"Solver status: {model['status']}")

if __name__ == "__main__":
    main()
