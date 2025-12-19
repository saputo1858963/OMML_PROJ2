import numpy as np
import functions_4_BIAtomici as fn 

def main():
    # Data loading
    X_train, Y_train, X_test, Y_test = fn.extract_data_Q4()
    
    # --- HYPERPARAMETER SELECTION (Grid Search) ---
    # We select only two classes to tune the parameters (e.g. 2 vs 3)
    # in order to use BinarySVM for the GridSearch
    mask_cv = (Y_train == 2) | (Y_train == 3)
    X_cv = X_train[mask_cv]
    y_cv = Y_train[mask_cv]
    # Conversion of labels in +1/-1 for the BinarySVM
    y_cv_bin = np.where(y_cv == 2, 1.0, -1.0)
    
    grid_C = [0.1, 1.0, 10.0, 100.0]
    grid_gamma = [1e-3, 1e-2, 1e-1, 1.0]
    
    # Grid Search
    gs_results = fn.grid_search_cv(
        X_cv, y_cv_bin, 
        grid_C=grid_C, 
        grid_gamma=grid_gamma, 
        k_fold=5,
        max_iter=300 
    )
    
    best_params = gs_results['best_params']
    C_val = best_params['C']
    gamma_val = best_params['gamma']
    
    # Initialization and training (Multiclass OAO)
    model = fn.MulticlassSVM_OAO(C=C_val, gamma=gamma_val)
    
    # Training
    model.fit(X_train, Y_train)
    
    # Prediction and metrics
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    
    acc_train = np.mean(Y_pred_train == Y_train)
    acc_test = np.mean(Y_pred_test == Y_test)
    
    # Confusion matrix
    unique_labels = np.unique(Y_train)
    cm = fn.compute_confusion_matrix(Y_test, Y_pred_test, unique_labels)
    
    print("\nThe used kernel is the: RBF")
    print(f"{'C':<30} {C_val}")
    print(f"{'Gamma':<30} {gamma_val}")
    print(f"{'Accuracy on training set':<30} {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"{'Accuracy on test set':<30} {acc_test:.4f} ({acc_test*100:.2f}%)")
    print(f"{'Run Time (seconds)':<30} {model.stats['time']:.4f}")
    print(f"{'Iterations (Cumulative)':<30} {model.stats['iterations']}")
    print(f"{'Gap m(alpha) - M(alpha)':<30} {model.stats['kkt_viol']:.6f}")
    print(f"{'Optimal value (Sum Dual Obj)':<30} {model.stats['dual_obj']:.4f}")
    print(f"Solver status: {model.stats.get('status', 'optimal')}")
    print(f"{'Multiclass Strategy':<30} One-Against-One (OAO)")
    
    print("-" * 30)
    fn.print_confusion_matrix_formatted(cm, unique_labels)
    print("-" * 30)

if __name__ == "__main__":
    main()