# DQEF

This is the implementation of "Unveiling Hidden Gems: Enhancing Entity Resolution with a Data Perspective"

## Environment

* Python 3.7
* PyTorch 1.4
* HuggingFace Transformers
* NLTK (for 1-N ER problem)

## Train

```
python train.py \ 
	--task Amazon \
	--batch_size 16 \
	--max_len 512 \
	--lr 1e-5 \
	--n_epochs 10 \
	--finetuning \
	--split \
	--lm bert
```

- `--task`: the name of the tasks (see `task.json`)
- `--batch_size`, `--max_len`, `--lr`, `--n_epochs`: the batch size, max sequence length, learning rate, and the number of epochs
- `--split`: whether to split the attribute, should always be turned on
- `--finetuning`: whether to finetune the LM, should always be turned on
- `--lm`: the language model. We now support `bert`, `distilbert`, `xlnet`, `roberta` (`bert` by default)
  - If you want to load the model file locally, you can configure the `--lm_path`
