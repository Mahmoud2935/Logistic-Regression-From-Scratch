from Models.logistic_regression import *
from preprocessing.preprocessing import * 
from  Metrics.metrics import accuracy
import pandas as pd
import random
# load data (اقرا البيانات وخزنها)


df = pd.read_csv("data/student_pass.csv")

x_data = df[["Hours","Attendance","Assignments"]]
y_data = df["Pass"]

# ليست ب نمباي  بحيث تقدر تعلم شافل وترين وتيست براحتك 
x_data = x_data.to_numpy().tolist()
y_data = y_data.to_list()

#shuffle
data = list(zip(x_data,y_data))
random.shuffle(data)
x_data,y_data= zip(*data)
x_data = list(x_data)
y_data = list(y_data)

# trian/test split
split_index = int(len(x_data) * 0.8)
x_train = x_data[:split_index]
x_test = x_data[split_index:]

y_train = y_data[:split_index]
y_test = y_data[split_index:]

#normalize train 
x_train,min_vals,max_vals = normalize_train(x_train)

# normalize test 
x_test = normalize_test(x_test,min_vals,max_vals)

w,b,losses  = trian(x_train,y_train)
# Save loss values for visualization outside the training pipeline.
import json
with open("experiments/losses.json", "w") as file:
    json.dump(losses, file)

# Generate probability predictions for test samples.
probability = []
for xi in x_test:
    probability.append(predction(xi,w,b))

# Convert probabilities into binary class predictions.
classes = classfy(probability,0.5)

# Evaluate model performance on the test set.
Accuracy = accuracy(classes, y_test)

# Analyze model predictions using a confusion matrix.
TP = TN = FP = FN = 0

for true, pred in zip(y_test, classes):

    if true == 1 and pred == 1:
        TP += 1

    elif true == 0 and pred == 0:
        TN += 1

    elif true == 0 and pred == 1:
        FP += 1

    elif true == 1 and pred == 0:
        FN += 1

# Display a summary of training and evaluation results.
print("\n" + "=" * 40)
print("Training Summary")
print("=" * 40)

print(f"Final Loss  : {losses[-1]:.4f}")
print(f"Train Size  : {len(x_train)}")
print(f"Test Size   : {len(x_test)}")
print(f"Accuracy    : {Accuracy:.2f}%")

print("\nConfusion Matrix")
print("-" * 20)
print(f"TN : {TN} | FP : {FP}")
print(f"FN : {FN} | TP : {TP}")

print("=" * 40)
