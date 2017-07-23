import numpy as np
import numpy.linalg as nla

def debug_initialize_weights(fan_out, fan_in):
    w = np.sin(np.arange(1,(fan_in+1)*fan_out + 1)) / 10 # fan_in+1 signifies addition of bias term +1 just to cover end term
    #return w.reshape(fan_out,fan_in+1, order='F') # matlab follows column major reshapring
    return w.reshape(fan_out,fan_in+1) # cancel that column major, all that matters is being consistent within the language you chose
    

# Need to implement more elegant way of computing with 
# two versions of cost function, with reg and without
def compute_numerical_gradient(cost_fn, theta, X,y, input_layer_size, 
                        hidden_layer_size, num_labels, lda): #theta is the unrolled nn_params
    numgrad = np.zeros(len(theta))
    perturb = np.zeros(len(theta))
    e = 1e-4
    for p in range(0, len(theta)):
        perturb[p] = e
        loss1 = cost_fn(theta-perturb, X,y, input_layer_size, 
                            hidden_layer_size, num_labels, lda)
        loss2 = cost_fn( theta+perturb, X,y,input_layer_size, 
                            hidden_layer_size, num_labels, lda)
        
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad
        
def check_nn_gradients(cost_fn, gradient_fn, lda = None):
    if lda == None:
        lda = 0

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing...
    X = debug_initialize_weights(m, input_layer_size-1)
    y = 1 + np.mod(np.arange(1,m+1),num_labels)
    
    # flatten the weight matrices in order to conform to the interface of the cost and
    # gradient backprop functions
    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    # compute gradient via backprop
    backpropgrad = gradient_fn(nn_params, X,y,input_layer_size, hidden_layer_size,num_labels, lda)
    
    # Compute numerical gradient
    numgrad = compute_numerical_gradient(cost_fn, nn_params, X, y,
                                         input_layer_size, hidden_layer_size,
                                         num_labels, lda)
    
    # compare the gradient and numerical gradient visually
    print("The analytically computed gradient (via backpropagation):\n",backpropgrad)
    print("The numerically computed gradient:\n", numgrad)
    #compute relative difference
    diff = nla.norm(numgrad-backpropgrad) / nla.norm(numgrad+backpropgrad)
    print("The relative difference between the numerical and analytical gradient is:",diff)
    
