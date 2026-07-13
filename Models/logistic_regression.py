import math
# Convert any value into a probability between 0 and 1
def sigmoid(x):
    return 1 / (1+math.exp(-x))

# Calculate the probability for a single sample
def predction(x,w,b):
    z = sum(xi*wi for xi,wi in zip(x,w))
    return sigmoid(z+b)

# Train the model by updating weights and bias
# using prediction error and Gradient Descent.
def trian(x_data,y_data,epoches=100,lr=0.1):
    # Store loss values to monitor training progress.
    losses = []

   #Initialize weights and bias with zeros.
    w = [0]* len(x_data[0])
    b = 0

    for epoch  in range(epoches):
        # Accumulate prediction errors for the current epoch.
        epoch_loss = 0

        for x,y in zip(x_data,y_data):
            y_pred = predction(x,w,b)
            # Calculate prediction error.
            error = y_pred-y
            # Add squared error to measure model loss.
            epoch_loss += error ** 2
          
            # Calculate bias gradient.
            db = error
             # Calculate gradients for all weights.
            dw = []
            for xi in x:
                dw.append(error*xi)

            for i in range(len(w)):
                # Update each weight using Gradient Descent.
                w[i] = w[i]-lr*dw[i]
           # Update bias using the prediction error.
            b = b -lr*db
           
        epoch_loss /= len(x_data)
        losses.append(epoch_loss)
            
    # Stop training when the loss becomes sufficiently small.  
        if epoch_loss < 0.06:
            print(f"Stopped at epoch {epoch+1}")
            break

    return w,b,losses


# Convert probabilities into binary class labels.
def classfy(prob,throsheld):
    classes = []
    for i in prob:
        if i >= throsheld:
          classes.append(1)
        else:
           classes.append(0)
    return classes