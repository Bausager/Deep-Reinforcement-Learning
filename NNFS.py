import numpy as np
import pickle
import copy


"""
Input layer for forward pass
"""
class Layer_Input:
    
    # Forwards pass
    def forward(self, inputs, training):
        self.output = inputs


"""
Dense layer
"""
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, 
                 weight_regularizer_L1=0, weight_regularizer_L2=0, 
                 bias_regularizer_L1=0, bias_regularizer_L2=0):
        
        # Initilizing weights
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initilizing biases
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2
    
    # Forward pass
    def forward(self, inputs, training):
        # Remember values
        self.inputs = inputs
        # Calculate output values from input, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backwards pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on regularization
        # L1 on weights
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        # L2 on weights
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights

        # L1 on biases
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        # L2 on biases
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases       
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
       
    # Retrive layer parameter
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set weights and biases in layer insatants
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
             
        
class Layer_Dropout:
    # Initilize
    def __init__(self, rate):
        # Store rate.
        # We invert it as for example for dropout
        # 0.1, we need a succes rate of 0.9
        self.rate = (1 - rate)
        
    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        
        # If not in th traning mode - return values
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)/self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask
        
    # Backwards pass
    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.binary_mask


   
# ReLU Activation
class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs, training):
        # Calculate output values from the input
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
        
    # Backwards pass
    def backward(self, dvalues):
        # Since we need to modify the original variale
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
    # Calculate predictions for output
    def predictions(self, outputs):
        return outputs
        
# Softmax Activation
class Activation_Softmax:
    
    # Forward pass
    def forward(self, inputs, training):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    # Backwards pass
    def backward(self, dvalues):
        
        # Creates uninitilized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) \
                - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    # Calculate predictions for output
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

"""
Sigmoid activation
"""
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs, training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    
    
# Linear activation
class Activation_Linear:
    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # Derivative is 1, 1 * dvalues = dvalues - the chian rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for output
    def predictions(self, outputs):
        return outputs



# Commen loss class
class Loss:
    
    # Calculates the data and regularization losses
    # given output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):
        
        # Calculate simple loss
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        # If just data losses
        if not include_regularization:
            return data_loss
        
        # Returns losses
        return data_loss, self.regularization_loss()
    
    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    # Regularization loss calculation
    def regularization_loss(self):
        
        # 0 by default
        regularization_loss = 0
        
        # Calculate regulazation loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
            
            # L1 regularization - weights
            # Calculate only when factor greater than 0
            if layer.weight_regularizer_L1 > 0:
                regularization_loss += layer.weight_regularizer_L1 \
                            * np.sum(np.abs(layer.weights))
                
            # L2 regularization - weights
            # Calculate only when factor greater than 0
            if layer.weight_regularizer_L2 > 0:
                regularization_loss += layer.weight_regularizer_L2 \
                            * np.sum(layer.weights * layer.weights)
    
            # L1 regularization - biases
            # Calculate only when factor greater than 0
            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 \
                            * np.sum(np.abs(layer.biases))
                
            # L2 regularization - biases
            # Calculate only when factor greater than 0
            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 \
                            * np.sum(layer.biases * layer.biases)    
                
                
            return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
# Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
     
# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples     
     
        
# Mean Squred Error loss
class Loss_MeanSquredError(Loss):
    # Forwards pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_loss = np.mean((y_true - y_pred)**2, axis=1)
        # Return losses
        return sample_loss
    
    # Backwards pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        # Number of output in every sample
        # We'll take the first sample to count
        outputs = len(dvalues[0])
        
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):
    # Forwards pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_loss = np.mean(np.abs(y_true - y_pred), axis=1)
        # Return losses
        return sample_loss
    
    # Backwards pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        # Number of output in every sample
        # We'll take the first sample to count
        outputs = len(dvalues[0])
        
        # Gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
        
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
        
    # Backwards pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        
        # If labels are on-hot encoded,
        # turn them into descreate values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate Gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize Gradient
        self.dinputs = self.dinputs/samples
    
    
    
class Optimizer_SGD:
    
    # Initialize optimizer - set settings
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        
    # Update parameters
    def update_params(self, layer):
        
        # If we use momentum
        if self.momentum:
            
            # If layer does not contaion momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # the array doesn't exist for biases yet either
                layer.bias_momentums = np.zeros_like(layer.biases)
        
            # Build weight updates with momntum - take previous
            # updates multiplied by retain factor and update with
            # current gradient
            weight_updates = self.momentum*layer.weight_momentums - self.current_learning_rate*layer.dweights
            layer.weight_momentums = weight_updates
            
            # Build bias updates
            bias_update = self.momentum*layer.bias_momentums - self.current_learning_rate*layer.dbiases
            layer.bias_momentums = bias_update
          
        # Vanilla SGD updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_update = -self.current_learning_rate * layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_update
    
        
    def post_update_params(self):
        self.iterations += 1
    
    
class Optimizer_Adagrad:
    
    # Initialize optimizer - set settings
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        
    # Update parameters
    def update_params(self, layer):
        
             
        # If layer does not contaion momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # If there is no momentum array for weights
            # the array doesn't exist for biases yet either
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        # Vanilla SGD parameter update + normilization
        # with squre rooted cache
        layer.weights += -self.current_learning_rate*layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
        
    def post_update_params(self):
        self.iterations += 1
 
    
class Optimizer_RMSProp:
    
    # Initialize optimizer - set settings
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        
    # Update parameters
    def update_params(self, layer):
        
             
        # If layer does not contaion momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # If there is no momentum array for weights
            # the array doesn't exist for biases yet either
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + (1.0 - self.rho) * layer.dweights**2
        layer.bias_cache =  self.rho * layer.bias_cache + (1.0 - self.rho) * layer.dbiases**2
        
        # Vanilla SGD parameter update + normilization
        # with squre rooted cache
        layer.weights += -self.current_learning_rate*layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
        
    def post_update_params(self):
        self.iterations += 1
    
       
class Optimizer_Adam:
    
    # Initialize optimizer - set settings
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        
    # Update parameters
    def update_params(self, layer):
        
        # If layer does not contaion momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            # If there is no momentum array for weights
            # the array doesn't exist for biases yet either
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # Get corrected momentum 
        # self.iteration is 0 at first pass
        # and we need to start woth 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))
        # Update cache with squred cuurent gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1.0 - self.beta_2) * layer.dweights**2
        layer.bias_cache =  self.beta_2 * layer.bias_cache + (1.0 - self.beta_2) * layer.dbiases**2
        
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))
        
        # Vanilla SGD parameter update + normilization
        # with squre rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
            
        
    def post_update_params(self):
        self.iterations += 1
       
    
 

  # Model class
class Model:
    def __init__(self, *, N_Step_TD=None, N_states=6, N_actions=2, gamma=0.99, RL_batch_size=1024):
        # Create a list of network ojects
        self.layers = []
        # Softmax classifier's output oject
        self.softmax_classifier_output = None

        # list of printouts
        self.history_epochs = []
        self.history_steps = []
        self.history_accuracy = []
        self.history_loss = []
        self.history_data_loss = []
        self.history_regularization_loss = []
        self.history_learning_rate = []

        # Variables for Reinforcement Learning
        if N_Step_TD is not None:
            self.N_Step_TD = int(N_Step_TD)
            self.RL_batch_size = RL_batch_size
            self.discount = np.logspace(0, N_Step_TD+1, num=N_Step_TD+1, 
                                        base=gamma, endpoint=False)
            self.steps = 0
            self.batch_steps = 0
            self.TD_rewards = np.zeros([self.RL_batch_size, N_Step_TD+1])
            self.Calc_TD_reward = np.zeros([self.RL_batch_size])
            self.TD_state = np.zeros([self.RL_batch_size, N_states])
            self.q_values = np.zeros([self.RL_batch_size, N_actions])
            self.actions = np.zeros([self.RL_batch_size, 1])


        
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    # Set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
    
    def RL_train(self, X, *, reward=0):

        action = -1

        if self.steps == 0:
            self.TD_state[self.batch_steps,:] = np.array([X])
            self.q_values[self.batch_steps,:] = self.predict(X)[0,:]
            #print(f'Predection: {self.q_values[self.batch_steps,:]}')
            self.actions[self.batch_steps] = (np.argmax(self.q_values[self.batch_steps,:]))
            #print(f'Action: {self.actions[self.batch_steps]}')

            self.steps += 1


        elif(self.steps % int(self.N_Step_TD+1)) == 0:
            self.TD_rewards[self.batch_steps, self.steps-1] = np.max(self.predict(X))
            self.Calc_TD_reward[self.batch_steps] = np.sum(self.TD_rewards[self.batch_steps, :]*self.discount) 
            
            self.q_values[self.batch_steps, int(self.actions[self.batch_steps])] = self.Calc_TD_reward[self.batch_steps]

            self.batch_steps += 1
            if self.batch_steps >= self.RL_batch_size:
                self.train(X=self.TD_state, y=self.q_values, print_every=None)
                self.batch_steps = 0

            self.steps = 0

        else:
            self.TD_rewards[self.batch_steps, self.steps-1] = reward[0]
            self.steps += 1      


        if np.random.rand() > 0.35:
            action = np.argmax(self.predict(X)[0,:])
            #print(f'Predection: {self.predict(X)[0,:]}')
        else:
            action = np.random.randint(len(self.predict(X)[0,:]))      

        return action


# Train the model
    def train(self, X, y, *, 
                epochs=1, 
                batch_size=None,
                print_every='epoch', 
                validation_data=None, 
                validation_split=None,
                history='epoch'):

        if validation_data != None and validation_split != None:
            raise ValueError("Ether choose validation_data OR validation_split")


        # Initialize accuracy object
        self.accuracy.init(y)
        # Default value if batch size is not being set
        train_steps = 1
                    
        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

        if history=='step':
            x = 0


            # Main training loop
        for epoch in range(1, epochs+1):
            # Print epoch number
            #print(f'epoch: {epoch}')
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            # Iterate over steps
            
            for step in range(train_steps):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                    
            
                # Perform the forward pass
                output = self.forward(batch_X, training=True)
                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)
                # Perform backward pass
                self.backward(output, batch_y)
                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                

                if print_every == None:
                    pass
                elif print_every != 'epoch':
                    if not step % print_every and step != 0:
                        print(f'epoch: {epoch}, ' +
                              f'step: {step}, ' +
                              f'acc: {accuracy:.3f}, ' +
                              f'loss: {loss:.3f} (' +
                              f'data_loss: {data_loss:.3f}, ' +
                              f'reg_loss: {regularization_loss:.3f}), ' +
                              f'lr: {self.optimizer.current_learning_rate:.6f}')
                else:
                    if step == train_steps -1:
                        print(f'epoch: {epoch}, ' +
                              f'acc: {accuracy:.3f}, ' +
                              f'loss: {loss:.3f} (' +
                              f'data_loss: {data_loss:.3f}, ' +
                              f'reg_loss: {regularization_loss:.3f}), ' +
                              f'lr: {self.optimizer.current_learning_rate:.6f}')
                    
                if step == train_steps -1 and history=='epoch':
                    self.history_epochs.append(epoch)
                    self.history_accuracy.append(accuracy)
                    self.history_loss.append(loss)
                    self.history_data_loss.append(data_loss)
                    self.history_regularization_loss.append(regularization_loss)
                    self.history_learning_rate.append(self.optimizer.current_learning_rate)
                elif history=='step':
                    x = x + 1
                    self.history_steps.append(x)
                    self.history_accuracy.append(accuracy)
                    self.history_loss.append(loss)
                    self.history_data_loss.append(data_loss)
                    self.history_regularization_loss.append(regularization_loss)
                    self.history_learning_rate.append(self.optimizer.current_learning_rate)

        # Get and print epoch loss and accuracy
        epoch_data_loss, epoch_regularization_loss = \
             self.loss.calculate_accumulated(
                 include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()

        if print_every is not None:
            print('Training:  ' +
                   f'acc: {epoch_accuracy:.3f}, ' +
                   f'loss: {epoch_loss:.3f} (' +
                   f'data_loss: {epoch_data_loss:.3f}, ' +
                   f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                   f'lr: {self.optimizer.current_learning_rate:.6f}')
            
        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data != None or validation_split != None:
             self.evaluate(*validation_data, batch_size=batch_size)


    # Evaluates the model using passed-in data
    def evaluate(self, X_val, y_val, *, batch_size=None):
        
        # Default value if batch size is not set
        validation_steps = 1
        
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice into batches
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
                
            # Perform the forward pass
            output = self.forward(batch_X, training=False)
            
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            
            # Get prediction and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
        
        # Get and print validatio and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        
        # Print a summery
        print('Validation: ' +
              f'acc: {validation_accuracy:.3f}, ' + 
              f'loss: {validation_loss:.3f}.')
        
        

    # Finalize the model
    def finalize(self):
        
        # Create and set the input layer
        self.input_layer = Layer_Input()
        # Count all ojects
        layer_count = len(self.layers)
        
        # Initialize list of trainable layers
        self.trainable_layers = []
        
        # Iterate the ojects
        for i in range(layer_count):
            
            # If it's first layer,
            # the previous layer oject is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
                
            # The last layer - the next oject is the loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases - 
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                    self.trainable_layers.append(self.layers[i])
                    
        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
        
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossentropy):
                # Create an object of combined activation
                # and loss functions
                self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
            

                
    # Performs forwards pass
    def forward(self, X, training):
        
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" oject is expecting
        self.input_layer.forward(X, training)
        
        # Call forward method of every oject in a chain
        # Pass output od the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
            
        # "layer" is now the last object from the list,
        # return its output
        return layer.output
    
    # Backwards pass
    def backward(self, output, y):
        
        # If softmax classifier
        if self.softmax_classifier_output is not None:   
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            
            # Since we'll not call backward method of the last layer
            # which is softmax activition
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs
                
            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        # First call backward methon on the loss
        # this will set dinputs property that the last
        # layer will try to acces shortly
        self.loss.backward(output, y)
        
        
        # Call backward mehton going through all the ojects
        # in reverse order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    
    # Predicts on the sample
    def predict(self, X, *, batch_size=None):
        
        # Default value if batch size is not being set
        prediction_steps = 1
        
        # Calculates number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps*batch_size < len(X):
                prediction_steps += 1
                
        # Model output
        output = []
        
        # Iterate over steps
        for step in range(prediction_steps):
            
            # If batch size is not set
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X
            
            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
                
            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            
            # Append batch prediction to the list of predictions
            output.append(batch_output)
            
            # Stack and return results
            return np.vstack(output)
            
        
    # Retrives and returns parameters of trainable layers
    def get_parameters(self):
        
        # Creates a list for parameters 
        parameters = []
        
        # Iterate trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
            
        # Return a list
        return parameters
    

    # Updates the model with new parameters
    def set_parameters(self, parameters):
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    # Save the parameters into a file
    def save_parameters(self, path):
        
        # Open a files in the binary-write mode
        # and save parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
            
    # Load the parameters into a file
    def load_parameters(self, path):
        
        # Open a files in the binary-read mode
        # and load parameters into trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
            
    # Saves the model
    def save(self, path):
        # Make a deep copy of current model instants
        model = copy.deepcopy(self)
        
        # Reset accumualation values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        # Remove data from input layer
        # and gratients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        # For each layer remove inputs, outputs and dinputs properties
        for layer in model.layers:
            for property in ['input', 'output', 'dinput', 'dwieghts', 'dbiases']:
                layer.__dict__.pop(property, None)
                
        # Open a file in binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            
    # Loads and returns model
    @staticmethod
    def load(path):
        
        # Open file with binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        # Return a model
        return model

    
    
# Commen accuracy class
class Accuracy:
    
    # Calculate an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        
        # Get comparison results
        comparisons = self.compare(predictions, y)
        
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        # Calculate accuracy
        accuracy = np.mean(comparisons)
        
        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        # Return the data and regularization losses
        return accuracy
    
    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    
# Accuracy calculatioins for regresion model
class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        # Create precision property
        self.precision = None
        
    # Calculates precision value
    # based on passed-in grund truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    # Compares predection to the ground truth values
    def compare(self, predections, y):
        return np.absolute(predections - y) < self.precision
    
# Accuracy calculations for classification
class Accuracy_Categorical(Accuracy):
    
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
        
    # No initialization is needed
    def init(self, y):
        pass
    
    # Compares prediction to the ground truth values
    def compare(self, predections, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predections == y
    
