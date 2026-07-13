# Calculate classification accuracy by comparing predicted classes
# with the true labels, then return the percentage of correct predictions.

def accuracy(classes, true_lable):
    corrected_count = 0
    for i in range(len(classes)):
        if classes[i] == true_lable[i]:
            corrected_count += 1
    return corrected_count / len(classes) * 100



