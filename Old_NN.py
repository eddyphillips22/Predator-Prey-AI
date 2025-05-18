import numpy as np
import os

np.set_printoptions(suppress=True, precision=8)

def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of the sigmoid function
    return x * (1 - x)

# Training data: [distance_to_prey, predator_hunger]
X = np.array([
    [1, 0], [0.5, 0.2], [0.5, 0.8], [0.1, 1], [0.7, 0.2], [0.9, 0.1]
])

y = np.array([
    [1], [1], [0], [0], [1], [1]
])

input_size = 2
hidden_size = 5
output_size = 1

np.random.seed(42)

def ai_train():
    print("Current working directory:", os.getcwd())

    # Load or initialize weights for input-to-hidden layer.
    if os.path.exists("weights_input_hidden.csv") and os.path.getsize("weights_input_hidden.csv") > 0:
        weights_input_hidden = np.loadtxt("weights_input_hidden.csv", delimiter=",")
        if weights_input_hidden.ndim == 1:
            if weights_input_hidden.size == 0:
                weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
            else:
                weights_input_hidden = weights_input_hidden.reshape(input_size, hidden_size)
        if weights_input_hidden.shape != (input_size, hidden_size):
            weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
    else:
        weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1


    # Load or initialize weights for hidden-to-output layer.
    if os.path.exists("weights_hidden_output.csv") and os.path.getsize("weights_hidden_output.csv") > 0:
        weights_hidden_output = np.loadtxt("weights_hidden_output.csv", delimiter=",")
        # If the loaded array is 1D (e.g. shape is (5,)) then reshape it.
        if weights_hidden_output.ndim == 1:
            # Check if the loaded array is not empty
            if weights_hidden_output.size == 0:
                weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
            else:
                weights_hidden_output = weights_hidden_output.reshape(hidden_size, output_size)
        # If shape doesn't match, reinitialize.
        if weights_hidden_output.shape != (hidden_size, output_size):
            weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
    else:
        weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1

    # Training loop.
    for iteration in range(10000):
        hidden_input = np.dot(X, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, weights_hidden_output)
        final_output = sigmoid(final_input)
        
        output_error = y - final_output
        output_delta = output_error * sigmoid_derivative(final_output)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
        
        weights_hidden_output += hidden_output.T.dot(output_delta)
        weights_input_hidden += X.T.dot(hidden_delta)

    # Save weights to CSV.
    try:
        np.savetxt("weights_input_hidden.csv", weights_input_hidden, delimiter=",")
        np.savetxt("weights_hidden_output.csv", weights_hidden_output, delimiter=",")
        print("Weights saved to CSV files.")
    except Exception as e:
        print("Error saving weights:", e)
    
    # Verify by immediately loading the weights.
    try:
        wih_check = np.loadtxt("weights_input_hidden.csv", delimiter=",")
        who_check = np.loadtxt("weights_hidden_output.csv", delimiter=",")
        if who_check.ndim == 1:
            who_check = who_check.reshape(hidden_size, output_size)
        print("Verified weights_input_hidden shape:", wih_check.shape)
        print("Verified weights_hidden_output shape:", who_check.shape)
    except Exception as e:
        print("Error loading weights after saving:", e)
    
    return final_output

def ai_run(normalised_distance, normalised_hunger):
    if os.path.exists("weights_input_hidden.csv") and os.path.getsize("weights_input_hidden.csv") > 0:
        weights_input_hidden = np.loadtxt("weights_input_hidden.csv", delimiter=",")
        if weights_input_hidden.ndim == 1:
            if weights_input_hidden.size == 0:
                weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
            else:
                weights_input_hidden = weights_input_hidden.reshape(input_size, hidden_size)
        if weights_input_hidden.shape != (input_size, hidden_size):
            weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
    else:
        weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1

        
    if os.path.exists("weights_hidden_output.csv") and os.path.getsize("weights_hidden_output.csv") > 0:
        weights_hidden_output = np.loadtxt("weights_hidden_output.csv", delimiter=",")
        # If the loaded array is 1D (e.g. shape is (5,)) then reshape it.
        if weights_hidden_output.ndim == 1:
            # Check if the loaded array is not empty
            if weights_hidden_output.size == 0:
                weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
            else:
                weights_hidden_output = weights_hidden_output.reshape(hidden_size, output_size)
        # If shape doesn't match, reinitialize.
        if weights_hidden_output.shape != (hidden_size, output_size):
            weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
    else:
        weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
        
    input_vector = np.array([[normalised_distance, normalised_hunger]])
    
    hidden_input = np.dot(input_vector, weights_input_hidden)            # Shape: (1, hidden_size)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output)             # Shape: (1, output_size)
    final_output = sigmoid(final_input)
    return final_output[0, 0]

# --- Prey neural net configuration ---
prey_input_size  = 2   # e.g. [norm_pred_dist, norm_hunger]
prey_hidden_size = 5
prey_output_size = 1

def prey_train():
    """
    Similar to ai_train, but uses its own datasets (X_prey, y_prey)
    and saves weights to prey_weights_input_hidden.csv / prey_weights_hidden_output.csv.
    """
    # Example training data: [norm_pred_dist, norm_hunger] → [flee(1)/stay(0)]
    # Replace these with your real prey training examples later.
    X_prey = np.array([
        [0.0, 1.0],  # predator on you and you're starving → definitely flee
        [1.0, 0.0],  # predator far away and you're full → definitely stay
        [0.5, 1.0],  # predator half-range and hungry → maybe flee
        [0.8, 0.2],  # far predator, not hungry → probably ignore
    ])
    y_prey = np.array([[1],[0],[1],[0]])

    # load or init weights
    if os.path.exists("prey_weights_input_hidden.csv") and os.path.getsize("prey_weights_input_hidden.csv")>0:
        wih = np.loadtxt("prey_weights_input_hidden.csv", delimiter=",")
        if wih.ndim==1: wih = wih.reshape(prey_input_size, prey_hidden_size)
        if wih.shape!=(prey_input_size,prey_hidden_size):
            wih = 2*np.random.random((prey_input_size,prey_hidden_size))-1
    else:
        wih = 2*np.random.random((prey_input_size,prey_hidden_size))-1

    if os.path.exists("prey_weights_hidden_output.csv") and os.path.getsize("prey_weights_hidden_output.csv")>0:
        who = np.loadtxt("prey_weights_hidden_output.csv", delimiter=",")
        if who.ndim==1: who = who.reshape(prey_hidden_size, prey_output_size)
        if who.shape!=(prey_hidden_size, prey_output_size):
            who = 2*np.random.random((prey_hidden_size,prey_output_size))-1
    else:
        who = 2*np.random.random((prey_hidden_size,prey_output_size))-1

    # simple training loop
    for _ in range(5000):
        h_in  = np.dot(X_prey, wih)
        h_out = sigmoid(h_in)
        f_in  = np.dot(h_out, who)
        f_out = sigmoid(f_in)

        err = y_prey - f_out
        delta_out = err * sigmoid_derivative(f_out)
        err_h    = delta_out.dot(who.T)
        delta_h  = err_h * sigmoid_derivative(h_out)

        who += h_out.T.dot(delta_out)
        wih += X_prey.T.dot(delta_h)

    # save
    np.savetxt("prey_weights_input_hidden.csv",  wih, delimiter=",")
    np.savetxt("prey_weights_hidden_output.csv", who, delimiter=",")
    return f_out

def prey_run(norm_pred_dist, norm_hunger):
    """
    Inference for prey: returns a 0–1 “flee” score.
    """
    # load weights (same pattern as ai_run, but different files)
    wih = (np.loadtxt("prey_weights_input_hidden.csv", delimiter=",")
        .reshape(prey_input_size, prey_hidden_size))
    who = (np.loadtxt("prey_weights_hidden_output.csv", delimiter=",")
        .reshape(prey_hidden_size, prey_output_size))

    inp = np.array([[norm_pred_dist, norm_hunger]])  # shape (1,2)
    h   = sigmoid(np.dot(inp, wih))
    out = sigmoid(np.dot(h,  who))
    return out[0,0]


prey_train()




