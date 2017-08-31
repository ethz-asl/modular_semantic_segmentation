install:
	pip install -r requirements.txt
	pip install -e .
	sh setup.sh

test:
	pytest .
	flake8
