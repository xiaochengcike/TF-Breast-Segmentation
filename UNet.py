import tensorflow as tf
import numpy as np

class TensorFlowUNet:
	def __init__(dataset, dropout_rate = 0.25):
		self.dataset = dataset
		self.dropout_rate = dropout_rate

		self.weight_dict = self.initialize_weights()
		self.bias_dict = self.initialize_bias()

	def model_inference(self, x):
		pass

	def tf_variable(self, shape):
		return tf.Variable(tf.random_normal(shape))

	def convolution(self, x, weight, bias, stride):
		convolved = tf.nn.conv2d(x, weight, [1, stride, stride, 1], 'SAME')
		bias_convolved = tf.nn.bias_add(convolved, bias)
		return tf.nn.relu(bias_convolved)

	def max_pooling(self, x):
		return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	def convolution_transpose(self, x, weight, bias, shape):
		deconvolved = tf.nn.conv2d_transpose(x, weight, shape, [1, 2, 2, 1])
		bias_deconvolved = tf.nn.bias_add(deconvolved, bias)
		return tf.nn.relu(bias_deconvolved)

	def copy_and_crop(self, left_tensor, right_tensor):
		if left_tensor.shape == right_tensor.shape
			return tf.concat([left_tensor, right_tensor], 3)
		else:
			 raise Exception('Left Tensor', left_tensor, ' does not equal the shape of Right Tensor', right_tensor)

	def double_convolution(self, x, weight1, bias1, weight2, bias2):
		conv1 = self.convolution(x, weight1, bias1, 3)
		conv2 = self.convolution(conv1, weight2, bias2, 3)
		return tf.nn.dropout(conv2, self.dropout_rate)