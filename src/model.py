import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

np.random.seed(1) 

def show(array, msg='default message'):
    plt.figure(figsize=(16, 8))

    # Plot the original image
    plt.subplot(3, 4, 1)
    plt.title(msg)
    #print(f'original: {x_test[total-1]}')
    plt.imshow(array[0], cmap='gray')
    plt.colorbar()
    plt.axis('off')

    for filter_ind in range(1, len(array)): 

    # Plot the normalized convolved output
        plt.subplot(3, 4, filter_ind + 1)
        plt.title(f"Filter {filter_ind}")
        plt.imshow(array[filter_ind], cmap='gray')
        plt.colorbar()
        plt.axis('off')
    plt.show()

def show_pool(array, msg='default message'):
    plt.figure(figsize=(16, 8))

    # Plot the original image
    plt.subplot(3, 4, 1)
    plt.title(msg)
    #print(f'original: {x_test[total-1]}')
    plt.imshow(array[0], cmap='gray')
    plt.colorbar()
    plt.axis('off')

    for filter_ind in range(1, len(array)): 

    # Plot the normalized convolved output
        plt.subplot(3, 4, filter_ind + 1)
        plt.title(f"Pool {filter_ind}")
        plt.imshow(array[filter_ind], cmap='gray')
        plt.colorbar()
        plt.axis('off')
    plt.show()





class CNN:
    '''
    CNN model made by Sunoo Kim (DGIST) 
    \ntodo:
    - maxpooling
    - add layer with activation function
    - convolve (stride, pooling, no padding, relu)
    - batch
    - hidden layer's size
    - conv filter size
    - number of filter
    - train time
    \nconstraints:
    - no padding
    - filter should be square
    '''

    def __init__(self):
        '''
        initializations
        '''
        self.layers = ['initial'] # 'initial' will be replaced by single batch input
        self.filters = ['initial'] 
        self.operations = ['initial'] # ('conv','convolve','maxpooling','flatten','normal','normal')
        self.weights = ['initial']
        self.activations = ['initial']
        self.output_shapes = ['initial']
        self.derivatives = ['initial']

        self.delta=1e-11


    def im2col(self, latest_layer, input_shape, stride, filter_size, target_shape, pad=0):
        filter_h, filter_w = filter_size
        C, H, W = latest_layer.shape
        latest_layer = latest_layer.reshape(1, C, H, W)

        # Apply padding
        padded_input = np.pad(latest_layer, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

        # Create sliding windows
        windows = sliding_window_view(padded_input, (C, filter_h, filter_w), axis=(1, 2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :, :]

        # Reshape to 2D
        N, C, out_h, out_w, _, _, _ = windows.shape
        col = windows.reshape(N, C, out_h, out_w, -1).transpose(0, 2, 3, 1, 4).reshape(N * out_h * out_w, -1)
        
        return col


    def col2im(self, col, input_shape, filter_size, stride=1):
        channels, height, width = input_shape
        if isinstance(filter_size, int):
            filter_height, filter_width = filter_size, filter_size
        else:
            filter_height, filter_width = filter_size

        # Calculate output spatial dimensions
        out_height = (height - filter_height) // stride + 1
        out_width = (width - filter_width) // stride + 1

        # Initialize the output image
        image = np.zeros((channels, height, width), dtype=col.dtype)

        # Reshape col back to sliding window regions
        col = col.reshape(channels, filter_height, filter_width, out_height, out_width)
        col = col.transpose(0, 3, 4, 1, 2)  # Rearrange to (channels, out_height, out_width, filter_height, filter_width)

        # Create the grid of indices for height and width
        i0 = np.repeat(np.arange(filter_height), filter_width).reshape(-1, 1)
        i1 = stride * np.repeat(np.arange(out_height), out_width).reshape(1, -1)
        j0 = np.tile(np.arange(filter_width), filter_height).reshape(-1, 1)
        j1 = stride * np.tile(np.arange(out_width), out_height).reshape(1, -1)

        # Calculate all indices
        i = i0 + i1  # Shape: (filter_height * filter_width, out_height * out_width)
        j = j0 + j1  # Shape: (filter_height * filter_width, out_height * out_width)
        k = np.repeat(np.arange(channels), i.shape[0]).reshape(-1, 1)  # Shape: (channels * filter_height * filter_width, 1)

        # Flatten col to match indexing
        col_flat = col.reshape(channels * filter_height * filter_width, -1)

        # Use np.add.at to map col values back to the image
        for channel in range(channels):
            np.add.at(
                image[channel], 
                (i.flatten(), j.flatten()), 
                col_flat[channel * filter_height * filter_width:(channel + 1) * filter_height * filter_width].flatten()
            )

        return image



    
    def forward(self):
        '''
        train self.single_batch_x -> (1, 28, 28)
        '''
        for index, operation in enumerate(self.operations):
            if index==0: continue
            if operation == 'conv':
                latest_layer = np.copy(self.layers[index-1]) # right before layer (1, 28, 28)
                # (1, 16, 25, 25)
                im2col_target_shape = (self.filters[index].shape[1], self.filters[index].shape[-1], self.layers[index].shape[1], self.layers[index].shape[1]) # will be (1, 16, 25, 25)
                target_shape = self.layers[index].shape
                filter_size = int(im2col_target_shape[1]**(1/2)) # 4
                after_conv_size = int(target_shape[2]) # 25
                stride = (len(latest_layer[0]) - filter_size) // (after_conv_size - 1) # 1
                print('latest layer shape:',latest_layer.shape)
                im2col_layer = self.im2col(latest_layer=latest_layer, input_shape=latest_layer.shape, stride=stride, filter_size=(filter_size, filter_size), target_shape=im2col_target_shape)
                im2col_layer = im2col_layer.reshape(im2col_target_shape[0] * im2col_target_shape[1], im2col_target_shape[2] * im2col_target_shape[3])

                self.layers[index] = np.matmul(self.filters[index].reshape(-1, im2col_layer.shape[0]), im2col_layer)
                self.layers[index] = self.layers[index].reshape(target_shape)           
                show(self.layers[index], 'after convolve') # test whether convolve is OK
                
                if self.activations[index] == 'relu':
                    self.layers[index] = self.relu(self.layers[index])


            elif operation == 'maxpooling':
                latest_layer = self.layers[index-1] # 8, 25, 25
                target_shape = self.layers[index].shape # 8, 12, 12
                im2col_shape = (len(latest_layer), self.filters[index][0] * self.filters[index][1]) + target_shape[-2:] # 8, 4, 12, 12
                stride = latest_layer.shape[-1] // target_shape[-1]
                self.layers[index] = self.im2col(latest_layer=latest_layer, input_shape=latest_layer.shape, stride=stride, filter_size=(stride, stride), target_shape=im2col_shape)

                # for backpropagation save argmax index
                tmp = np.transpose(self.layers[index],(0, 3, 1, 2)).reshape(-1, im2col_shape[1])
                arg_max = np.argmax(tmp, axis=1)
                self.derivatives[index] = arg_max
                
                self.layers[index] = np.max(self.layers[index], axis=1) # 8, 12, 12
                #show_pool(self.layers[index], 'after pooling')

            elif operation == 'flatten':
                self.layers[index] = self.layers[index-1].reshape(1, -1)
                
            elif operation == 'normal':
                '''
                calculate the weights with latest layer, activation
                '''
                self.layers[index] = np.matmul(self.layers[index-1], self.weights[index])
                if self.activations[index] == 'relu':
                    self.layers[index] = self.relu(self.layers[index])
                elif self.activations[index] == 'softmax':
                    self.layers[index] = self.softmax(self.layers[index])
                
            else:
                return Exception

    
    def backward(self):
        '''
        
        '''
        for index, operation in enumerate(self.operations[::-1]):
            real_index = -(index+1)
            if operation == 'normal' and index==0: # assume that the last layer is normal layer with softmax
                # softmax + cee gradient -> output layer - correct layer
                self.derivatives[real_index] = self.layers[real_index] - self.single_batch_y                
                
            elif operation == 'normal':
                self.weights[real_index+1] -= np.matmul(self.layers[real_index].T, self.derivatives[real_index+1]) * self.lr
                self.derivatives[real_index] = np.matmul(self.derivatives[real_index+1], np.maximum(self.weights[real_index+1], 0).T)
                
            elif operation == 'flatten':
                self.weights[real_index+1] -= np.matmul(self.layers[real_index].T, self.derivatives[real_index+1]) * self.lr
                self.derivatives[real_index] = np.matmul(self.derivatives[real_index+1], np.maximum(self.weights[real_index+1], 0).T) #
                
            elif operation == 'maxpooling':
                arg_max = self.derivatives[real_index]
                derivative = np.zeros((self.layers[real_index].size, self.filters[real_index][0]**2))
                row = np.arange(derivative.shape[0])
                derivative[row, arg_max] = self.derivatives[real_index+1].T.reshape(-1,) # 1152, 
                self.derivatives[real_index] = self.col2im(col=derivative, input_shape=self.layers[real_index-1].shape, filter_size=self.filters[real_index], stride=self.filters[real_index][0])
                #show(self.derivatives[real_index])
            elif operation == 'conv':
                # delta: calculated gradient from current layer (num_filters, H', W')
                # input_data: ouput of prev layer (channel, H, W)
                # filters:  (num_filters, filter_h, filter_w)
                delta = self.derivatives[real_index + 1]  

                relu_derivative = np.where(self.layers[real_index] > 0, 1, 0)
                delta *= relu_derivative  # Element-wise multiplication
                
                input_data = self.layers[real_index - 1] 
                filters = self.filters[real_index] 
                filter_height, filter_width = int(filters.shape[2]**(1/2)), int(filters.shape[2]**(1/2))
                stride = 1  

                input_shape = input_data.shape 
                output_shape = delta.shape 

                col_shape = (input_shape[0], filter_height* filter_width, output_shape[1], output_shape[2])
                input_cols = self.im2col(input_data, input_shape, stride, (filter_height, filter_width), col_shape)

                delta_flat = delta.reshape(delta.shape[0], -1)  
                input_cols = np.transpose(input_cols, (2, 3, 0, 1)).reshape(-1, input_cols.shape[0] * input_cols.shape[1])
                filter_gradients = np.matmul(delta_flat, input_cols)
                filter_gradients = filter_gradients.reshape(filters.shape)  

                self.filters[real_index] -= self.lr * filter_gradients

                flipped_filters = np.flip(filters, axis=(1, 2)) 
                flipped_filters_cols = flipped_filters.reshape(filters.shape[0], -1).T 

                prev_delta_cols = np.dot(flipped_filters_cols, delta_flat)  
                prev_delta = self.col2im(prev_delta_cols.T, input_shape, (filter_height, filter_width), stride)

                # save delta
                self.derivatives[real_index] = prev_delta

    
    def relu(self, matrix):
        return np.maximum(0, matrix)
    
    def softmax(self, matrix):
        exp_matrix = np.exp(matrix + self.delta)
        return exp_matrix / (np.sum(exp_matrix) + self.delta)
    
    def add_normal_layer(self, layer_size=64, act_func=None):
        '''
        act_func: 'relu', 'softmax'
        '''
        self.operations.append('normal')
        self.weights.append(np.random.randn(self.layers[-1].shape[1], layer_size) * 0.01) 
        self.layers.append(np.zeros((1, layer_size))) # 수정 요망
        self.filters.append(None)
        self.activations.append(act_func)
        self.derivatives.append(None)

    def add_conv_layer(self, 
                        num_filters=8, 
                        filter_size = (4, 4), 
                        input_shape = None, 
                        act_func=None, 
                        stride=1
                        ):
        '''
        act_func : 'relu'
        input_shape is required only for the first layer
        '''
        self.operations.append('conv')
        self.activations.append(act_func)
        self.weights.append(None)
        
        height = None
        width = None
        
        if input_shape is not None:
            # output height, width
            height = (input_shape[1] - filter_size[0])//stride + 1
            width = (input_shape[2] - filter_size[1])//stride + 1
            self.layers.append(np.zeros((num_filters, height, width)))
            self.filters.append(np.random.randn(num_filters, input_shape[0], filter_size[0] * filter_size[1]))
            self.derivatives.append(None)
        else:
            input_shape = self.layers[-1].shape
            height = (input_shape[1] - filter_size[0])//stride + 1
            width = (input_shape[2] - filter_size[1])//stride + 1
            self.layers.append(np.zeros((num_filters, height, width)))
            self.filters.append(np.random.randn(num_filters, input_shape[0], filter_size[0] * filter_size[1]))
            self.derivatives.append(None)
            pass # yet conv layer will be only one

    def addMaxPooling(self, pooling_size = (2, 2)): # no stride for pooling!!!! stride must be same as pooling_size
        '''
        
        '''
        # latest layer shape : (1, 8, 25, 25)
        self.operations.append('maxpooling')
        latest_shape = (self.layers[-1].shape)
        target_shape = (latest_shape[0], latest_shape[2] // pooling_size[0], latest_shape[2] // pooling_size[1])
        self.layers.append(np.zeros(target_shape))
        self.filters.append(pooling_size)
        self.activations.append(None)
        self.weights.append(None)
        self.derivatives.append(None)

    def flatten(self):
        '''
        flatten the array of layer right before this call.
        '''
        latest_layer_shape = self.layers[-1].shape
        self.operations.append('flatten')
        self.layers.append(np.zeros((1, latest_layer_shape[0] * latest_layer_shape[1] * latest_layer_shape[2])))
        self.filters.append(None)
        self.activations.append(None)
        self.weights.append(None)
        self.derivatives.append(None)
    
    def fit(self, 
        x_train : np.ndarray, 
        y_train : np.ndarray, 
        epochs : int,
        learning_rate = 1e-3, 
        batch_size = 500, 
        validation_split = 0.1,
        save_path = '../ckpt/ckpt{}.pkl'
        ):

        '''
        train
        '''
        print(f'train start\nbatch size: {batch_size}\nlearning rate: {learning_rate}')
        # initial
        self.lr = learning_rate
        x_train, y_train = np.array(x_train), np.array(y_train)

        # random shuffle for validation split
        s = np.arange(x_train.shape[0])
        np.random.shuffle(s)
        x_train, y_train = x_train[s], y_train[s]

        # train set, validation test split
        validation_index = int(len(x_train) * validation_split)
        self.validation_size = validation_index
        self.train_size = len(x_train) - self.validation_size
        self.validation_x, self.validation_y = x_train[-validation_index:], y_train[-validation_index:]
        self.x, self.y = x_train[:-validation_index], y_train[:-validation_index]
        
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}/{epochs}')

            s = np.arange(self.x.shape[0])
            np.random.shuffle(s)
            self.x, self.y = self.x[s], self.y[s]
            num_of_batchs = self.train_size // batch_size

            time_start = time.time()
            for batch_num in range(1, num_of_batchs+1):
                # Batch data
                start_index = (batch_num - 1) * batch_size
                end_index = batch_num * batch_size
                self.batch_x, self.batch_y = self.x[start_index:end_index], self.y[start_index:end_index]

                # Train on each data
                for i in range(len(self.batch_x)):
                    self.single_batch_x = self.batch_x[i].reshape(1, *self.batch_x[i].shape)
                    self.single_batch_y = self.batch_y[i]

                    # Assign input to the initial layer
                    self.layers[0] = self.single_batch_x
                    self.forward()
                    self.backward()

                # Calculate batch loss and accuracy
                batch_predictions = []
                batch_true_labels = []
                loss = 0
                for i in range(len(self.batch_x)):
                    self.single_batch_x = self.batch_x[i].reshape(1, *self.batch_x[i].shape)
                    self.layers[0] = self.single_batch_x
                    self.forward()

                    batch_predictions.append(np.argmax(self.layers[-1]))  # Prediction
                    batch_true_labels.append(np.argmax(self.batch_y[i]))  # True label
                    loss += -np.mean(np.log(self.layers[-1] + self.delta) * self.batch_y[i])
                loss /= len(self.batch_x)
                accuracy = np.mean(np.array(batch_predictions) == np.array(batch_true_labels))

                # Validation loss and accuracy
                val_predictions = []
                val_true_labels = []
                val_loss = 0
                for i in range(len(self.validation_x)):
                    self.single_batch_x = self.validation_x[i].reshape(1, *self.validation_x[i].shape)
                    self.layers[0] = self.single_batch_x
                    self.forward()

                    val_predictions.append(np.argmax(self.layers[-1]))
                    val_true_labels.append(np.argmax(self.validation_y[i]))
                    val_loss += -np.mean(np.log(self.layers[-1] + self.delta) * self.validation_y[i])

                val_loss /= len(self.validation_x)
                val_accuracy = np.mean(np.array(val_predictions) == np.array(val_true_labels))

                # Print progress
                load_time = round(time.time() - time_start, 2)
                time_per_step = round(load_time * 1000 / num_of_batchs)

                prefix = f'{batch_num}/{num_of_batchs}'
                accuracy *= 100
                val_accuracy *= 100
                suffix = f'- {round(time.time()-time_start, 2)}s {time_per_step}ms/step - loss: {loss:.4f} - accuracy:{accuracy:.2f}% - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.2f}%'
                fill = '█'
                length = 30
                percent = f"{100 * (batch_num / float(num_of_batchs)):.1f}"
                filled_length = int(length * batch_num // num_of_batchs)
                bar = fill * filled_length + '-' * (length - filled_length)
                print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')

                if batch_num == num_of_batchs:
                    print()
                    print()
            weights_filename = save_path.format('')
            self.save_weights(weights_filename)
    
    def evaluate(self, x_test, y_test):
        predictions = []
        true_labels = []

        # Forward pass for all test data
        for i in range(len(x_test)):
            test_input = x_test[i].reshape(1, *x_test[i].shape)  # Ensure proper input shape
            self.layers[0] = test_input
            self.forward()

            predictions.append(np.argmax(self.layers[-1]))  # Predicted label
            true_labels.append(np.argmax(y_test[i]))  # True label
        
        # Calculate accuracy
        test_accuracy = np.mean(np.array(predictions) == np.array(true_labels)) * 100

        return test_accuracy
    def save_weights(self, filepath):
        data = {
            'weights': self.weights,
            'filters': self.filters,
            'operations': self.operations,
            'activations': self.activations
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Weights and filters saved to {filepath}")
    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.filters = data['filters']
        self.operations = data['operations']
        self.activations = data['activations']
        print(f"Weights and filters loaded from {filepath}")

