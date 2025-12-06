# Main script for Question 2 - Group BIAtomici

import numpy as np
import functions_2_BIAtomici as f

def main():
    print("##############################")
    print("Processing Question 2")
    print("##############################")

    # 1. Load Data
    X_train, Y_train, X_test, Y_test = f.extract_data()

    # 2. Hyperparameters (MUST BE SAME AS Q1)
    # REPLACE THESE with the values you found in Q1
    best_C = 10     # Example value
    best_gamma = 0.01  # Example value
    
    # 3. Decomposition Settings
    q_value = 4  # Working set size (Must be even, >= 4)
    
    # 4. Train with Decomposition Method
    print(f"Training SVM using Decomposition Method (q={q_value})...")
    alpha_opt, b_opt, stats = f.solve_svm_decomposition(
        X_train, Y_train, best_C, best_gamma, q=q_value
    )
    
    # 5. Predictions
    y_train_pred = f.predict(X_train, Y_train, alpha_opt, b_opt, X_train, best_gamma)
    y_test_pred = f.predict(X_train, Y_train, alpha_opt, b_opt, X_test, best_gamma)
    
    # 6. Metrics
    train_acc, cm_train = f.compute_metrics(Y_train, y_train_pred)
    test_acc, cm_test = f.compute_metrics(Y_test, y_test_pred)

    # 7. Print Outputs
    print(f"The used kernel is the: RBF")
    print(f"C                                     {best_C}")
    print(f"Gamma                                 {best_gamma}")
    print(f"Value of q                            {q_value}")
    print(f"Accuracy on training set              {train_acc:.4f} %")
    print(f"Accuracy on test set                  {test_acc:.4f} %")
    print(f"Run Time (seconds)                    {stats['time']:.4f}")
    print(f"Iterations (Outer)                    {stats['iterations']}")
    print(f"KKT violations (m(a) - M(a))          {stats['kkt_gap']:.6f}")
    print(f"Starting value                        0.0")
    print(f"Optimal value                         {stats['final_obj']:.6f}")
    print(f"Solver status                         {stats['status']}")
    
    print("\nConfusion Matrix (Test Set):")
    print(f"                 Pred Pullover   Pred Dress")
    print(f"True Pullover    {cm_test[0,0]}              {cm_test[0,1]}")
    print(f"True Dress       {cm_test[1,0]}              {cm_test[1,1]}")

if __name__ == "__main__":
    main()