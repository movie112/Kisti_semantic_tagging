0. python3.6 라이브러리 설정
	* tensorflow==1.15.0
    * torch==1.8.1
	* transformers==4.7.0

1. tf 모델 준비

```
./
├── convert_tf_checkpoint_to_pytorch_BERT.py
├── model
│   ├── bert_config_kisti.json
│   ├── model.ckpt-262500.data-00000-of-00001
│   ├── model.ckpt-262500.index
│   └── model.ckpt-262500.meta
├── pytorch_model.bin
└── test.py
```

2. tf 모델을 torch 버전으로 컨버팅

```
python convert_tf_checkpoint_to_pytorch_BERT.py
```

3. 컨버팅된 모델 결과이 로드되는지 확인

```
python test.py
```
