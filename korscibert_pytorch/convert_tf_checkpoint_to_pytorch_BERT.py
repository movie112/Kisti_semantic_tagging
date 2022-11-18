# https://github.com/woshiyyya/DFGN-pytorch/blob/master/DFGN/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py

"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
import tensorflow as tf
import torch
import numpy as np

# from .modeling import BertConfig, BertForPreTraining
# from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
# from transformers import ElectraConfig, ElectraPreTrainedModel

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
	config_path = os.path.abspath(bert_config_file)
	tf_path = os.path.abspath(tf_checkpoint_path)
	print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
	# Load weights from TF model
	init_vars = tf.train.list_variables(tf_path)
	names = []
	arrays = []
	for name, shape in init_vars:
		print("Loading TF weight {} with shape {}".format(name, shape))
		array = tf.train.load_variable(tf_path, name)
		names.append(name)
		arrays.append(array)

	# Initialise PyTorch model
	# config = ElectraConfig.from_json_file(bert_config_file)
	config = BertConfig.from_json_file(bert_config_file)

	print("Building PyTorch model from configuration: {}".format(str(config)))
	model = BertForPreTraining(config)
	# model = ElectraPreTrainedModel(config)

	out_name = ['current_loss_scale', '', '']
	for name, array in zip(names, arrays):
		name = name.split('/')
		# adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
		# which are not required for using pretrained model
		if any(n in ["adam_v", "adam_m", "current_loss_scale", "global_step", "good_steps"] for n in name):
			print("Skipping {}".format("/".join(name)))
			continue
		pointer = model
		for m_name in name:
			if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
				l = re.split(r'_(\d+)', m_name)
			else:
				l = [m_name]
			if l[0] == 'kernel' or l[0] == 'gamma':
				pointer = getattr(pointer, 'weight')
			elif l[0] == 'output_bias' or l[0] == 'beta':
				pointer = getattr(pointer, 'bias')
			elif l[0] == 'output_weights':
				pointer = getattr(pointer, 'weight')
			# mjo
			else:
			# elif l[0] == 'current_loss_scale':
				pointer = getattr(pointer, l[0])
				# pass

			if len(l) >= 2:
				num = int(l[1])
				pointer = pointer[num]
		if m_name[-11:] == '_embeddings':
			pointer = getattr(pointer, 'weight')
		elif m_name == 'kernel':
			array = np.transpose(array)
		try:
			assert pointer.shape == array.shape
		except AssertionError as e:
			e.args += (pointer.shape, array.shape)
			raise
		print("Initialize PyTorch weight {}".format(name))
		pointer.data = torch.from_numpy(array)

	# Save pytorch-model
	print("Save PyTorch model to {}".format(pytorch_dump_path))
	torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	## Required parameters
	# parser.add_argument("--tf_checkpoint_path",
	#					 default = None,
	#					 type = str,
	#					 required = True,
	#					 help = "Path the TensorFlow checkpoint path.")
	# parser.add_argument("--bert_config_file",
	#					 default = None,
	#					 type = str,
	#					 required = True,
	#					 help = "The config json file corresponding to the pre-trained BERT model. \n"
	#						 "This specifies the model architecture.")
	# parser.add_argument("--pytorch_dump_path",
	#					 default = None,
	#					 type = str,
	#					 required = True,
	#					 help = "Path to the output PyTorch model.")
	args = parser.parse_args()

	# args.tf_checkpoint_path = './input_bert_base/bert_model.ckpt'
	# args.bert_config_file = './input_bert_base/bert_config.json'
	# args.pytorch_dump_path = './output/pytorch_model.bin'


	# args.tf_checkpoint_path = './input_koscibert2/model.ckpt-262500'
	# args.bert_config_file = './input_koscibert2/bert_config_kisti.json'
	# args.pytorch_dump_path = './output/pytorch_model.bin'

	# args.tf_checkpoint_path = 'koscibert-15330/TF/model.ckpt-262500'
	# args.bert_config_file   = 'koscibert-15330/TF/bert_config_kisti.json'
	# args.pytorch_dump_path  = 'koscibert-15330/PT/pytorch_model.bin'
	
	args.tf_checkpoint_path = 'model/model.ckpt-262500'
	args.bert_config_file   = 'model/bert_config_kisti.json'
	args.pytorch_dump_path  = 'pytorch_model.bin'

	convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
									 args.bert_config_file,
									 args.pytorch_dump_path)
