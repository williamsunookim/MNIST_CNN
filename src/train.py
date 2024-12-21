from model import CNN
from data import MnistDataloader

training_images_filepath = '../dataset/train-images.idx3-ubyte'
training_labels_filepath = '../dataset/train-labels.idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath)
x_train, label, y_train = mnist_dataloader.load_data()

model = CNN()
model.add_conv_layer(num_filters=8, filter_size=(4, 4), input_shape=(1, 28, 28), act_func='relu', stride=1)
model.addMaxPooling()
model.flatten()
model.add_normal_layer(layer_size=64, act_func='relu')
model.add_normal_layer(layer_size=10, act_func='softmax')

epochs = 20
lr = 2e-4
model.fit(x_train=x_train, y_train=y_train, epochs=20, validation_split=0.1, learning_rate=lr)