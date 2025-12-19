import numpy as np
import functions_3_BIAtomici as f

def main():
    try:
        X_train, Y_train, X_test, Y_test = f.extract_data()
    except FileNotFoundError as e:
        print(e)
        return

    # HYPERPARAMETERS (values from Q1)
    best_C = 100     
    best_gamma = 0.01  
    
    alpha, b, stats = f.solve_svm_mvp(X_train, Y_train, best_C, best_gamma)
    
    # Predictions
    y_train_pred = f.predict(X_train, Y_train, alpha, b, X_train, best_gamma)
    y_test_pred = f.predict(X_train, Y_train, alpha, b, X_test, best_gamma)
    
    # Metrics
    train_acc, _ = f.compute_metrics(Y_train, y_train_pred)
    test_acc, cm_test = f.compute_metrics(Y_test, y_test_pred)

    print(f"C: {best_C}")
    print(f"Gamma: {best_gamma}")
    print(f"Value of q: 2")
    print(f"Accuracy on training set: {train_acc:.4f} %")
    print(f"Accuracy on test set: {test_acc:.4f} %")
    print(f"Run Time (seconds): {stats['time']:.4f}")
    print(f"Iterations: {stats['iterations']}")
    print(f"KKT violations (m(a) - M(a)): {stats['gap']:.6f}")
    print(f"Optimal dual function value: {stats['final_obj']:.6f}")
    print(f"Total computed column of Q: {stats['cols_computed']}") 
    print(f"Solver status: {stats['status']}")
    
    print("\nConfusion Matrix (Test Set):")
    print(f"                 Pred Pullover   Pred Dress")
    print(f"True Pullover    {cm_test[0,0]}              {cm_test[0,1]}")
    print(f"True Dress       {cm_test[1,0]}              {cm_test[1,1]}")

if __name__ == "__main__":
    main()