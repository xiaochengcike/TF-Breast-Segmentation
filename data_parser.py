import nibabel as nib
import numpy as np
import os

class DataParser:
	def __init__(self, data_dir, train_fraction = 0.8, random_batches = False, data_shape = (256, 256)):
		self.train_fraction = np.clip(train_fraction, 0.0, 1.0)
		self.scan = []
		self.gt = []
		self.num_samples = 0
		self.random_batches = random_batches

		print('Parsing dataset...')
		for (root, dirs, files) in os.walk(os.path.join(data_dir, 'Scans')):
			for file in files:
				try:
					if file.split('.')[-1] == 'nii':
						scan_matrix = nib.load(os.path.join(data_dir, 'Scans', file)).get_fdata()
						gt_matrix = nib.load(os.path.join(data_dir, 'GroundTruth', file)).get_fdata()
						if scan_matrix.shape == gt_matrix.shape and scan_matrix.shape[:2] == data_shape:
							self.scan += np.dsplit(scan_matrix, scan_matrix.shape[-1])
							self.gt += np.dsplit(gt_matrix, gt_matrix.shape[-1])
							self.num_samples += gt_matrix.shape[-1]
				except Exception as e:
					print(str(e))
		indices = np.random.permutation(len(self.gt))
		self.gt = np.array(self.gt)[indices]
		self.scan = np.array(self.scan)[indices]
		
		self.train_index = 0
		self.test_index = self.num_samples * self.train_fraction

		print('Found %d training instances - %d training and %d testing' % (self.num_samples, 0.8 * self.num_samples, 0.2 * self.num_samples))

	def get_next_train(self, batch_size, scan_buffer = None, gt_buffer = None):
		if self.random_batches:
			indices = np.random.choice(int(self.num_samples * self.train_fraction), batch_size)
			return self.scan[indices], self.gt[indices]
		else:
			if self.train_index + batch_size < len(self.num_samples) * self.train_fraction:
				tmp = self.train_index
				self.train_index += batch_size

				if scan_buffer is None:
					return self.scan[int(tmp):int(self.train_index)], self.gt[int(tmp):int(self.train_index)]
				else:
					return np.concatenate((self.scan[int(tmp):int(self.train_index)], scan_buffer), axis = 0), np.concatenate((self.gt[int(tmp):int(self.train_index)], gt_buffer), axis = 0)

			offset = (self.train_index + batch_size) - (len(self.num_samples) * self.train_fraction)
			tmp = self.train_index
			self.train_index = 0

			if scan_buffer is None:
				return self.get_next_train(offset, scan_buffer = self.scan[int(tmp):int(len(self.num_samples) * self.train_fraction)], gt_buffer = self.gt[int(tmp):int(len(self.num_samples) * self.train_fraction)])
			else:
				return self.get_next_train(offset, scan_buffer = np.concatenate((self.scan[int(tmp):int(len(self.num_samples) * self.train_fraction)], scan_buffer), axis = 0), gt_buffer = np.concatenate((self.gt[int(tmp):int(len(self.num_samples) * self.train_fraction)], gt_buffer), axis = 0))

	def get_next_test(self, batch_size, scan_buffer = None, gt_buffer = None):
		if self.random_batches:
			indices = np.random.choice(int(self.num_samples * (1 - self.train_fraction)), batch_size) + self.num_samples
			return self.scan[indices], self.gt[indices]
		else:
			if self.test_index + batch_size < len(self.num_samples):
				tmp = self.test_index
				self.test_index += batch_size

				if scan_buffer is None:
					return self.scan[int(tmp):int(self.test_index)], self.gt[int(tmp):int(self.test_index)]
				else:
					return np.concatenate((self.scan[int(tmp):int(self.test_index)], scan_buffer), axis = 0), np.concatenate((self.gt[int(tmp):int(self.test_index)], gt_buffer), axis = 0)

			offset = (self.test_index + batch_size) - len(self.num_samples)
			tmp = self.test_index
			self.test_index = len(self.num_samples) * self.train_fraction

			if scan_buffer is None:
				return self.get_next_test(offset, scan_buffer = self.scan[int(tmp):int(len(self.num_samples))], gt_buffer = self.gt[int(tmp):int(len(self.num_samples))])
			else:
				return self.get_next_test(offset, scan_buffer = np.concatenate((self.scan[int(tmp):int(len(self.num_samples))], scan_buffer), axis = 0), gt_buffer = np.concatenate((self.gt[int(tmp):int(len(self.num_samples))], gt_buffer), axis = 0))

if __name__ == '__main__':
	main()