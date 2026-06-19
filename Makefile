.PHONY: data env

env:
	python -m venv .venv
	.venv\Scripts\pip install -e .

data:
	.venv\Scripts\python src/data/download_aemo.py
	.venv\Scripts\python src/data/parse_aemo.py
	.venv\Scripts\python src/data/clean_data.py
	.venv\Scripts\python src/data/feature_engineering.py
	.venv\Scripts\python src/data/split_and_save.py
