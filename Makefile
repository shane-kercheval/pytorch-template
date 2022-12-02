####
# DOCKER
####
docker_build:
	docker compose -f docker-compose.yml build

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	docker compose -f docker-compose.yml build --no-cache

docker_bash:
	docker compose -f docker-compose.yml up --build bash

docker_open: notebook mlflow_ui zsh

notebook:
	open 'http://127.0.0.1:8888/?token=d4484563805c48c9b55f75eb8b28b3797c6757ad4871776d'

zsh:
	docker exec -it data-science-template-bash-1 /bin/zsh

docker_all:
	docker compose run --no-deps --entrypoint "make all" bash

####
# MLFLOW
####
mlflow_ui:
	open 'http://127.0.0.1:1235'

mlflow_kill:
	 pkill -f gunicorn

mlflow_clean:
	rm -rf mlruns
	rm -f mlflow.db
	rm -rf mlflow-artifact-root
	rm -rf mlflow_server/1235

####
# Project
####
linting:
	flake8 --max-line-length 99 source/domain
	flake8 --max-line-length 99 source/entrypoints
	flake8 --max-line-length 99 tests

tests: linting
	rm -f tests/test_files/log.log
	#python -m unittest discover tests
	#pytest tests
	coverage run -m pytest tests
	coverage html

open_coverage:
	open 'htmlcov/index.html'

data_extract:
	python source/entrypoints/cli.py extract

data: data_extract

pytorch_fully:
	jupyter nbconvert --execute --to html source/notebooks/pytorch_fully_connected.ipynb
	mv source/notebooks/pytorch_fully_connected.html output/pytorch_fully_connected.html

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: data tests remove_logs exploration experiments

## Delete all generated files (e.g. virtual)
clean: mlflow_clean
	rm -f data/raw/*.pkl
	rm -f data/raw/*.csv
	rm -f data/processed/*
