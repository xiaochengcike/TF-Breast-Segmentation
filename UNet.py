import tensorflow as tf
import numpy as np

class TensorFlowUNet:
	def __init__(num_layers, num_iterations, learning_rate, pad_convolutions, input_shape, dropout_rate = 0.25):
		self.variables = {}
		self.num_layers = num_layers
		self.num_iterations = num_iterations
		self.learning_rate = learning_rate
		self.dropout_rate = dropout_rate
		self.padding = 'VALID'
		if pad_convolutions:
			self.padding = 'SAME'

		self.down_weights = []
		self.down_bias = []
		self.up_weights = []
		self.up_bias = []

		self.initialize_variables(input_shape, 64, 2)

	def initialize_variables(input_shape, first_conv_depth, final_conv_depth):
		features_in = input_shape[-1]
		features_out = first_conv_depth

		for layer_index in range(self.num_layers):
			self.down_weights.append(self.create_variable([3, 3, features_in, features_out]))
			self.down_weights.append(self.create_variable([3, 3, features_out, features_out]))
			self.down_bias.append(self.create_variable([features_out]))
			self.down_bias.append(self.create_variable([features_out]))

			for layer_index in range(num_layers):
				down_weights.append([3, 3, features_in, features_out])
				down_weights.append([3, 3, features_out, features_out])
				down_bias.append([features_out])
				down_bias.append([features_out])

				if layer_index < self.num_layers - 1:
					depth = first_conv_depth * 2 ** (self.num_layers - layer_index - 1)
					self.up_weights.append(self.create_variable([2, 2, depth // 2, depth]))
					self.up_weights.append(self.create_variable([3, 3, depth, depth // 2]))
					self.up_weights.append(self.create_variable([3, 3, depth // 2, depth // 2]))
					self.up_bias.append(self.create_variable([depth // 2]))
					self.up_bias.append(self.create_variable([depth // 2]))
					self.up_bias.append(self.create_variable([depth // 2]))
				else:
					self.up_weights.append(self.create_variable([1, 1, first_conv_depth, final_conv_depth]))
					self.up_bias.append(self.create_variable([final_conv_depth]))
			
			features_in = features_out
			features_out *= 2

	def train(self, dataset, batch_size):
		for iteration in range(self.num_iterations):
			data_list, label_list = dataset.get_next_train(batch_size)
			for batch_num in range(batch_size):
				data, labels = data_list[batch_num], label_list[batch_num]

	def model_inference(self, x):
		conv_layers = []
		for layer_index in range(self.num_layers):
			x = self.double_convolution(x, self.down_weights[layer_index * 2:layer_index * 2 + 2], self.down_bias[layer_index * 2:layer_index * 2 + 2])
	
			if layer_index < self.num_layers - 1:
				conv_layers.append(x)
				x = self.max_pooling(conv)

		for layer_index in range(self.num_layers):
			if layer_index < self.num_layers - 1:
				crop_index = self.num_layers - (layer_index + 1)
				up_conv = self.convolution_transpose(x, self.up_weights[layer_index * 3], self.up_bias[layer_index * 3])
				cropped_conv = self.copy_and_crop(conv_layers[crop_index], up_conv) # TODO FIX PLS
				x = self.double_convolution(x, self.up_weights[layer_index * 3 + 1:layer_index * 3 + 3], self.up_bias[layer_index * 3 + 1:layer_index * 3 + 3])
			else:
				x = self.convolution(x, self.up_weights[-1], self.up_bias[-1], 1)

		return x

	def create_variable(self, variable_shape):
		return tf.Variable(tf.random_normal(shape))

	def convolution(self, x, weight, bias, stride):
		convolved = tf.nn.conv2d(x, weight, [1, stride, stride, 1], self.padding)
		bias_convolved = tf.nn.bias_add(convolved, bias)
		return tf.nn.relu(bias_convolved)

	def max_pooling(self, x):
		return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], self.padding)

	def convolution_transpose(self, x, weight, bias):
		old_shape = tf.shape(x)
		new_shape = [old_shape[0], old_shape[1] * 2, old_shape[2] * 2, old_shape[3] // 2]
		deconvolved = tf.nn.conv2d_transpose(x, weight, new_shape, [1, 2, 2, 1], padding = self.padding)
		bias_deconvolved = tf.nn.bias_add(deconvolved, bias)
		return tf.nn.relu(bias_deconvolved)

	def copy_and_crop(self, left_tensor, right_tensor):
		if left_tensor.shape != right_tensor.shape
			xpadding = (left_tensor.shape[1] - right_tensor.shape[1]) // 2
			ypadding = (left_tensor.shape[2] - right_tensor.shape[2]) // 2
			left_tensor = tf.slice(left_tensor, [0, xpadding, ypadding, 0], [-1, right_tensor.shape[1], right_tensor.shape[2], -1])
			
		return tf.concat([left_tensor, right_tensor], 3)
		
	def double_convolution(self, x, weights, biases):
		conv1 = self.convolution(x, weights[0], biases[0], 3)
		conv2 = self.convolution(conv1, weights[1], biases[1], 3)
		return tf.nn.dropout(conv2, self.dropout_rate)