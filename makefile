install:
	pip install -r requirements.txt
	pip install -e .
	sh setup.sh
	sh download_data.sh

test:
	pytest .
	flake8