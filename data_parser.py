import nibabel as nib
from tqdm import tqdm
import numpy as np
import os

class DataParser:
	def __init__(self, data_dir, train_fraction = 0.8):
		self.train_fraction = np.clip(train_fraction, 0.0, 1.0)
		self.scan = []
		self.gt = []
		self.ids = []

		for (root, dirs, files) in os.walk(os.path.join(data_dir, 'Scans')):
			for file in tqdm(files, "Parsing Scans"):
				try:
					if file.split('.')[-1] == 'nii':
						self.scan.append(nib.load(os.path.join(data_dir, 'Scans', file)).get_fdata())
						self.gt.append(nib.load(os.path.join(data_dir, 'GroundTruth', file)).get_fdata())
						self.ids.append(int(file.split('.')[0]))
				except Exception as e:
					print(str(e))
		
		indices = np.random.permutation(len(self.gt))
		self.gt = np.array(self.gt)[indices]
		self.scan = np.array(self.scan)[indices]
		
		self.train_index = 0
		self.test_index = len(self.ids) * self.train_fraction

	def get_next_train(self, batch_size, scan_buffer = None, gt_buffer = None):
		if self.train_index + batch_size < len(self.ids) * self.train_fraction:
			tmp = self.train_index
			self.train_index += batch_size

			if scan_buffer is None:
				return self.scan[int(tmp):int(self.train_index)], self.gt[int(tmp):int(self.train_index)]
			else:
				return np.concatenate((self.scan[int(tmp):int(self.train_index)], scan_buffer), axis = 0), np.concatenate((self.gt[int(tmp):int(self.train_index)], gt_buffer), axis = 0)

		offset = (self.train_index + batch_size) - (len(self.ids) * self.train_fraction)
		tmp = self.train_index
		self.train_index = 0

		if scan_buffer is None:
			return self.get_next_train(offset, scan_buffer = self.scan[int(tmp):int(len(self.ids) * self.train_fraction)], gt_buffer = self.gt[int(tmp):int(len(self.ids) * self.train_fraction)])
		else:
			return self.get_next_train(offset, scan_buffer = np.concatenate((self.scan[int(tmp):int(len(self.ids) * self.train_fraction)], scan_buffer), axis = 0), gt_buffer = np.concatenate((self.gt[int(tmp):int(len(self.ids) * self.train_fraction)], gt_buffer), axis = 0))

		

	def get_next_test(self, batch_size, scan_buffer = None, gt_buffer = None):
		if self.test_index + batch_size < len(self.ids):
			tmp = self.test_index
			self.test_index += batch_size

			if scan_buffer is None:
				return self.scan[int(tmp):int(self.test_index)], self.gt[int(tmp):int(self.test_index)]
			else:
				return np.concatenate((self.scan[int(tmp):int(self.test_index)], scan_buffer), axis = 0), np.concatenate((self.gt[int(tmp):int(self.test_index)], gt_buffer), axis = 0)

		offset = (self.test_index + batch_size) - len(self.ids)
		tmp = self.test_index
		self.test_index = len(self.ids) * self.train_fraction

		if scan_buffer is None:
			return self.get_next_test(offset, scan_buffer = self.scan[int(tmp):int(len(self.ids))], gt_buffer = self.gt[int(tmp):int(len(self.ids))])
		else:
			return self.get_next_test(offset, scan_buffer = np.concatenate((self.scan[int(tmp):int(len(self.ids))], scan_buffer), axis = 0), gt_buffer = np.concatenate((self.gt[int(tmp):int(len(self.ids))], gt_buffer), axis = 0))

if __name__ == '__main__':
	main()