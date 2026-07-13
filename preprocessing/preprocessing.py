import numpy as np 
# Normalize training data and return min/max values.

def normalize_train(x_train):

    x_train = np.array(x_train, dtype=float)

    min_vals = np.min(x_train, axis=0)
    max_vals = np.max(x_train, axis=0)

    ranges = max_vals -min_vals
    # Prevent division by zero for constant features.
    ranges[ranges == 0] = 1

    x_train_norm = (x_train - min_vals) / ranges

    return x_train_norm.tolist() ,min_vals,max_vals

# Normalize test data using training min/max values: to avoid Data Leakage.
# (بتوع الترين علشان مستخدمش بتوع التيست ويبقي كده الموديل عارف الامتحان   min,max values خت نفس ) 


def normalize_test(x_test,min_vals,max_vals):

    x_test = np.array(x_test, dtype=float)

    ranges = max_vals-min_vals
    # Prevent division by zero for constant features.
    ranges[ranges == 0] = 1

    x_test_norm = (x_test - min_vals) / ranges

    return x_test_norm.tolist() 
        
     




