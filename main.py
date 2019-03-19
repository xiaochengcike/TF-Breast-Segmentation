from ISPY import ISPY
from UNet import TensorFlowUNet
from PIL import Image
import numpy as np

def main():
	ispy = ISPY('Data/', random_batches = True)
	data, labels = ispy.get_next_train(60)
	for i in range(len(data)):
		Image.fromarray(np.hstack((data[i], labels[i]))).show()
		input()
	#unet = TensorFlowUNet(ispy

	#unet.train(iterations = 100, learning_rate = 0.001, dataset = ispy)

	#print(unet.accuracy)


if __name__ == '__main__':
	main()
