export_env:
	conda env export | grep -v "^prefix: " > environment.yml

train:
	python train.py

visualize:
	python src/dataset.py

test:
	python test.py