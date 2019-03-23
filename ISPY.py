from BaseDataset import AbstractDataset
import nibabel as nib
import numpy as np
import os

class ISPY(AbstractDataset):
	def __init__(self, data_dir, train_fraction = 0.8, random_batches = False, data_shape = (256, 256, 1)):
		super().__init__(train_fraction, random_batches, data_shape)
		self.parse_dataset(data_dir, data_shape)

	def parse_dataset(self, data_dir, data_shape):
		print('Parsing...')
		for (root, dirs, files) in os.walk(os.path.join(data_dir, 'Scans')):
			for file in files:
				try:
					if file.split('.')[-1] == 'nii':
						scan_matrix = nib.load(os.path.join(data_dir, 'Scans', file)).get_fdata()
						gt_matrix = nib.load(os.path.join(data_dir, 'GroundTruth', file)).get_fdata()
						if scan_matrix.shape == gt_matrix.shape and scan_matrix.shape[:2] == data_shape:
							for norm_scan, norm_gt in zip(np.dsplit(scan_matrix, scan_matrix.shape[-1]), np.dsplit(gt_matrix, gt_matrix.shape[-1])):
								self.data.append(np.reshape(norm_scan, data_shape))
								self.labels.append(np.reshape(norm_gt, data_shape))
								self.num_samples += 1
				except Exception as e:
					print(str(e))

		super().shuffle_dataset()
		self.train_index = 0
		self.test_index = self.num_samples * self.train_fraction
		print('Parsed.')

	def get_next_train(self, batch_size):
		return self.get_next_train(batch_size, None, None)

	def get_next_test(self, batch_size):
		return self.get_next_test(batch_size, None, None)

	def get_next_train(self, batch_size, scan_buffer = None, gt_buffer = None):
		if self.random_batches:
			indices = np.random.choice(int(self.num_samples * self.train_fraction), batch_size)
			return self.data[indices], self.labels[indices]
		else:
			if self.train_index + batch_size < len(self.num_samples) * self.train_fraction:
				tmp = self.train_index
				self.train_index += batch_size

				if scan_buffer is None:
					return self.data[int(tmp):int(self.train_index)], self.labels[int(tmp):int(self.train_index)]
				else:
					return np.concatenate((self.data[int(tmp):int(self.train_index)], scan_buffer), axis = 0), np.concatenate((self.labels[int(tmp):int(self.train_index)], gt_buffer), axis = 0)

			offset = (self.train_index + batch_size) - (len(self.num_samples) * self.train_fraction)
			tmp = self.train_index
			self.train_index = 0

			if scan_buffer is None:
				return self.get_next_train(offset, scan_buffer = self.data[int(tmp):int(len(self.num_samples) * self.train_fraction)], gt_buffer = self.labels[int(tmp):int(len(self.num_samples) * self.train_fraction)])
			else:
				return self.get_next_train(offset, scan_buffer = np.concatenate((self.data[int(tmp):int(len(self.num_samples) * self.train_fraction)], scan_buffer), axis = 0), gt_buffer = np.concatenate((self.labels[int(tmp):int(len(self.num_samples) * self.train_fraction)], gt_buffer), axis = 0))

	def get_next_test(self, batch_size, scan_buffer = None, gt_buffer = None):
		if self.random_batches:
			indices = np.random.choice(int(self.num_samples * (1 - self.train_fraction)), batch_size) + self.num_samples
			return self.data[indices], self.labels[indices]
		else:
			if self.test_index + batch_size < len(self.num_samples):
				tmp = self.test_index
				self.test_index += batch_size

				if scan_buffer is None:
					return self.data[int(tmp):int(self.test_index)], self.labels[int(tmp):int(self.test_index)]
				else:
					return np.concatenate((self.data[int(tmp):int(self.test_index)], scan_buffer), axis = 0), np.concatenate((self.labels[int(tmp):int(self.test_index)], gt_buffer), axis = 0)

			offset = (self.test_index + batch_size) - len(self.num_samples)
			tmp = self.test_index
			self.test_index = len(self.num_samples) * self.train_fraction

			if scan_buffer is None:
				return self.get_next_test(offset, scan_buffer = self.data[int(tmp):int(len(self.num_samples))], gt_buffer = self.labels[int(tmp):int(len(self.num_samples))])
			else:
				return self.get_next_test(offset, scan_buffer = np.concatenate((self.data[int(tmp):int(len(self.num_samples))], scan_buffer), axis = 0), gt_buffer = np.concatenate((self.labels[int(tmp):int(len(self.num_samples))], gt_buffer), axis = 0))