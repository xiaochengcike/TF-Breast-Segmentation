import numpy as np
import abc

class AbstractDataset(abc.ABC):
	def __init__(self, train_fraction, random_batches, data_shape):
		super().__init__()
		self.train_fraction = np.clip(train_fraction, 0.0, 1.0)
		self.num_samples = 0
		self.random_batches = random_batches
		self.data_shape = data_shape

		self.data = []
		self.labels = []

	@abc.abstractmethod
	def parse_dataset(self, data_dir, data_shape):
		pass

	@abc.abstractmethod
	def get_next_train(self, batch_size):
		pass

	@abc.abstractmethod
	def get_next_test(self, batch_size):
		pass

	def shuffle_dataset(self):
		indices = np.random.permutation(len(self.labels))
		self.labels = np.array(self.labels)[indices]
		self.data = np.array(self.data)[indices]