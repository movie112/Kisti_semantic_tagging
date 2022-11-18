import torch
import transformers

model_path = 'pytorch_model.bin'

config = transformers.BertConfig.from_pretrained('model/bert_config_kisti.json')
# model = transformers.BertModel.from_pretrained(model_path, config=config) ## without cls
model = transformers.BertForPreTraining.from_pretrained(model_path, config=config) ## with cls 
print(model)
