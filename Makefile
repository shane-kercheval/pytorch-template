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

docker_open: notebook zsh

notebook:
	open 'http://127.0.0.1:8888/?token=d4484563805c48c9b55f75eb8b28b3797c6757ad4871776d'

zsh:
	docker exec -it data-science-template-bash-1 /bin/zsh

docker_all:
	docker compose run --no-deps --entrypoint "make all" bash


####
# Project
####
linting:
	ruff source/domain
	ruff source/entrypoints
	ruff tests

tests: linting
	rm -f tests/test_files/log.log
	#python -m unittest discover tests
	#pytest tests
	coverage run -m pytest tests
	coverage html

open_coverage:
	open 'htmlcov/index.html'


run_fc_1:
	python source/entrypoints/cli.py run \
		-config_file=source/entrypoints/run_fc_1.yaml \
		-device=cuda

run_cnn_1:
	python source/entrypoints/cli.py run \
		-config_file=source/entrypoints/run_cnn_1.yaml \
		-device=cuda

sweep_fc_1_bayes:
	python source/entrypoints/cli.py sweep \
		-config_file=source/entrypoints/sweep_fc_1_bayes.yaml \
		-device=cpu \
		-count=70

sweep_cnn_2_bayes:
	python source/entrypoints/cli.py sweep \
		-config_file=source/entrypoints/sweep_cnn_2_bayes.yaml \
		-device=cuda \
		-count=70

sweep_cnn_3:
	python source/entrypoints/cli.py sweep \
		-config_file=source/entrypoints/sweep_cnn_3.yaml \
		-device=cuda \
		-count=90

num_combinations:
	python source/entrypoints/cli.py num-combinations \
		-config_file=source/entrypoints/sweep_cnn_4.yaml

pytorch_fully:
	jupyter nbconvert --execute --to html source/notebooks/pytorch_fully_connected.ipynb
	mv source/notebooks/pytorch_fully_connected.html output/pytorch_fully_connected.html

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: data tests remove_logs pytorch_fully

## Delete all generated files (e.g. virtual)
clean:
	rm -f data/raw/*.pkl
	rm -f data/raw/*.csv
	rm -f data/processed/*
