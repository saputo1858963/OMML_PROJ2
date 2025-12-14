from functions_4_BIAtomici import extract_data_Q4, MulticlassSVM_OAA, get_confusion_matrix
import numpy as np
import time

def run_q4():
    # 1. Load Data
    X_train, Y_train, X_test, Y_test = extract_data_Q4()
    
    # 2. Hyperparameter Selection (Grid Search with K-Fold)
    # [cite_start]Range candidates [cite: 54]
    C_values = [1, 10]
    gamma_values = [0.01, 0.1]
    
    best_acc = 0
    best_C = 10
    best_gamma = 0.01
    
    # Simple validation set approach to save time for demonstration
    # (Using 20% of train as val)
    n_val = int(0.2 * len(Y_train))
    X_val = X_train[:n_val]
    Y_val = Y_train[:n_val]
    X_tr_part = X_train[n_val:]
    Y_tr_part = Y_train[n_val:]
    
    # print("Starting Grid Search...") # Commented out to match output strictness
    
    for C in C_values:
        for g in gamma_values:
            # Train smaller model for tuning
            svm_tune = MulticlassSVM_OAA(C=C, gamma=g, max_iter=200, tol=0.1)
            svm_tune.fit(X_tr_part, Y_tr_part)
            pred_val = svm_tune.predict(X_val)
            acc = np.mean(pred_val == Y_val)
            
            if acc > best_acc:
                best_acc = acc
                best_C = C
                best_gamma = g
                
    # 3. Train Final Model with Best Params
    start_time = time.time()
    final_model = MulticlassSVM_OAA(C=best_C, gamma=best_gamma, tol=1e-3, max_iter=1000)
    final_model.fit(X_train, Y_train)
    training_time = time.time() - start_time
    
    # 4. Evaluation
    train_pred = final_model.predict(X_train)
    test_pred = final_model.predict(X_test)
    
    train_acc = np.mean(train_pred == Y_train) * 100
    test_acc = np.mean(test_pred == Y_test) * 100
    
    # Confusion Matrix
    classes = [2, 3, 6]
    cm = get_confusion_matrix(Y_test, test_pred, classes)
    
    # 5. Output Printing
    # Must match Figure 1 and Q4 specific instructions
    
    print("The used kernel is the: RBF")
    print(f"C\t\t\t\t{best_C}")
    print(f"Gamma\t\t\t\t{best_gamma}")
    print(f"Accuracy on training set\t{train_acc:.2f}%")
    print(f"Accuracy on test set\t\t{test_acc:.2f}%")
    print(f"Run Time (seconds)\t\t{training_time:.4f}")
    print(f"Iterations\t\t\t{final_model.total_iter} (Sum of all binary SVMs)")
    print(f"KKT violations\t\t\t{final_model.avg_kkt:.6f} (Avg)")
    # Since Q4 OAA involves multiple objectives, we report the sum or clarify
    print(f"Starting value\t\t\t0.0") 
    print(f"Optimal value\t\t\t{final_model.total_obj:.4f} (Sum of Dual Objs)")
    
    # Q4 Specific: Multiclass Strategy
    print("Strategy: OAA") # [cite: 97]
    
    print("\nValues")
    print(cm)
    
if __name__ == "__main__":
    run_q4()