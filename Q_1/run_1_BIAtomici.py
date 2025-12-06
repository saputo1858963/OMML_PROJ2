# Main script for Question 1 - Group BIAtomici

import numpy as np
import functions_1_BIAtomici as f
import time

def main():
    print("##############################")
    print("Processing Question 1")
    print("##############################")

    # 1. Load Data
    X_train, Y_train, X_test, Y_test = f.extract_data()

    # 2. Grid Search (Comment out if you want to use fixed values to save time)
    # best_C, best_gamma = f.grid_search(X_train, Y_train, k=5)
    
    # Hardcoded for demonstration speed (replace with grid_search call above if needed)
    # Assuming Grid Search found C=10, Gamma=0.01 (Example values)
    best_C = 10
    best_gamma = 0.01 
    
    # 3. Train Final Model
    print("Training SVM with Best Hyperparameters...")
    alpha_opt, b_opt, stats = f.solve_svm_dual(X_train, Y_train, best_C, best_gamma)
    
    # 4. Predictions
    y_train_pred = f.predict(X_train, Y_train, alpha_opt, b_opt, X_train, best_gamma)
    y_test_pred = f.predict(X_train, Y_train, alpha_opt, b_opt, X_test, best_gamma)
    
    # 5. Metrics
    train_acc, cm_train = f.compute_metrics(Y_train, y_train_pred)
    test_acc, cm_test = f.compute_metrics(Y_test, y_test_pred)
    
    # Compute KKT Violation (Difference m(a) - M(a))
    # We need Q matrix one last time for this check
    K = f.rbf_kernel(X_train, X_train, best_gamma)
    Q = np.outer(Y_train, Y_train) * K
    kkt_gap = f.calculate_kkt_gap(alpha_opt, Q, Y_train, best_C)

    # 6. Print Outputs (Strictly following Figure 1/Instructions)
    print(f"The used kernel is the: RBF")
    print(f"C                                     {best_C}")
    print(f"Gamma                                 {best_gamma}")
    print(f"Accuracy on training set              {train_acc:.4f} %")
    print(f"Accuracy on test set                  {test_acc:.4f} %")
    print(f"Run Time (seconds)                    {stats['time']:.4f}")
    print(f"Iterations                            {stats['iterations']}")
    print(f"KKT violations (m(a) - M(a))          {kkt_gap:.6f}")
    print(f"Starting value                        0.0") # We started from zero
    print(f"Optimal value                         {stats['final_obj']:.6f}")
    print(f"Solver status                         {stats['status']}")
    
    print("\nConfusion Matrix (Test Set):")
    # Pretty print confusion matrix
    print(f"                 Pred Pullover   Pred Dress")
    print(f"True Pullover    {cm_test[0,0]}              {cm_test[0,1]}")
    print(f"True Dress       {cm_test[1,0]}              {cm_test[1,1]}")

if __name__ == "__main__":
    main()