import pickle
from data import MnistDataloader
from model import CNN


training_images_filepath = '../dataset/t10k-images.idx3-ubyte'
training_labels_filepath = '../dataset/t10k-labels.idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath)
x_test, label, y_test = mnist_dataloader.load_data()

# 모델 초기화
model = CNN()
model.add_conv_layer(num_filters=8, filter_size=(4, 4), input_shape=(1, 28, 28), act_func='relu', stride=1)
model.addMaxPooling()
model.flatten()
model.add_normal_layer(layer_size=64, act_func='relu')
model.add_normal_layer(layer_size=10, act_func='softmax')


# 저장된 가중치 로드
model.load_weights("../ckpt/ckpt.pkl")

# 평가
test_accuracy = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {test_accuracy:.2f}%")
