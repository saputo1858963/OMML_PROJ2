import numpy as np
import functions_2_BIAtomici as f

def main():
    # 1. Load Data
    try:
        X_train, Y_train, X_test, Y_test = f.extract_data()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Hyperparameters (set the best one of Q_1)
    best_C = 100       
    best_gamma = 0.01 
    
    # Decomposition parameter (Must be even, >= 4)
    q_value = 4       

    # 3. Train
    alpha, b, stats = f.solve_decomposition(X_train, Y_train, best_C, best_gamma, q=q_value)
    
    # 4. Predict
    y_train_pred = f.predict(X_train, Y_train, alpha, b, X_train, best_gamma)
    y_test_pred = f.predict(X_train, Y_train, alpha, b, X_test, best_gamma)
    
    # 5. Metrics
    train_acc, _ = f.compute_metrics(Y_train, y_train_pred)
    test_acc, cm_test = f.compute_metrics(Y_test, y_test_pred)

    print(f"C: {best_C}")
    print(f"Gamma: {best_gamma}")
    print(f"Value of q: {q_value}")
    print(f"Accuracy on training set: {train_acc:.4f} %")
    print(f"Accuracy on test set: {test_acc:.4f} %")
    print(f"Run Time (seconds): {stats['time']:.4f}")
    print(f"Iterations (Outer): {stats['iterations']}")
    print(f"KKT violations (m(a) - M(a)): {stats['gap']:.6f}")
    print(f"Optimal dual value: {stats['final_obj']:.6f}")
    print(f"Solver status: {stats['status']}")
    print(f"Number of function evaluations: {stats['func_eval']}")

    print("\nConfusion Matrix (Test Set):")
    print(f"                 Pred Pullover   Pred Dress")
    print(f"True Pullover    {cm_test[0,0]}              {cm_test[0,1]}")
    print(f"True Dress       {cm_test[1,0]}              {cm_test[1,1]}")

if __name__ == "__main__":
    main()