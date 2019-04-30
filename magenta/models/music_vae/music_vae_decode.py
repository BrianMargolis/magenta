from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from magenta import music as mm
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs


class VAEDecoder():
	def __init__(self, model_path=None, config_name=None, output_dir=None, max_batch_size=8,
	             config_map=configs.CONFIG_MAP):
		if config_name not in config_map:
			raise ValueError('Invalid config name: %s' % config_name)
		self.config = config_map[config_name]
		self.config.data_converter.max_tensors_per_item = None
		self.output_dir = os.path.expanduser(output_dir)

		checkpoint_path = os.path.expanduser(model_path)

		self.model = TrainedModel(
			self.config, batch_size=max_batch_size,
			checkpoint_dir_or_path=checkpoint_path)

		self.z_size = self.model._config.hparams.z_size

	def decode(self, z: np.ndarray):
		if len(z.shape) == 2 and z.shape[1] == self.z_size:
			# z is a single latent space
			midi = self._decode(z)
			return midi
		else:
			raise ValueError("z has inappropriate shape: {0}.\n"
			                 "Should be: (n_samples, {0})"
			                 .format(", ".join(z.shape), self.z_size))

	def _decode(self, z):
		return self.model.decode(z, length=self.config.hparams.max_seq_len)

	def _write(self, midi):
		date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
		basename = os.path.join(
			self.output_dir,
			'{}-*-{:03d}.mid'
				.format(date_and_time, len(midi)))
		for i, m in enumerate(midi):
			mm.sequence_proto_to_midi_file(m, basename.replace('*', '{:03d}'.format(i)))
