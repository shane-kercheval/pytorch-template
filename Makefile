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
	ruff source/library
	ruff cli.py
	ruff tests

tests: linting
	rm -f tests/test_files/log.log
	#python -m unittest discover tests
	#pytest tests
	coverage run -m pytest tests
	coverage html

open_coverage:
	open 'htmlcov/index.html'


# template / default runs
run_fc_default:
	python cli.py run -config_file=templates/run_fc.yaml

run_cnn_default:
	python cli.py run -config_file=templates/run_cnn.yaml


# sweep fc
sweep_fc:
	# run the latest config file
	python cli.py sweep \
		-config_file=$$(ls experiments/sweeps/sweep_fc_*.yaml | sort -V | tail -n 1)
		# using grid search via config setting
		# -runs=70 \

# sweep cnn
sweep_cnn:
	python cli.py sweep \
		-config_file=$$(ls experiments/sweeps/sweep_cnn_*.yaml | sort -V | tail -n 1) \
		-runs=70


# run fc
run_fc:
	python cli.py run \
		-config_file=$$(ls experiments/runs/run_fc_*.yaml | sort -V | tail -n 1)

# run cnn
run_cnn:
	python cli.py run \
		-config_file=$$(ls experiments/runs/run_cnn_*.yaml | sort -V | tail -n 1)


# number of combinations in default sweep configs
num_combinations_fc_default:
	python cli.py num-combinations \
		-config_file=source/entrypoints/default_sweep_fc.yaml

num_combinations_cnn_default:
	python cli.py num-combinations \
		-config_file=source/entrypoints/default_sweep_cnn.yaml


# number of combinations in latest sweep configs
num_combinations_fc:
	python cli.py num-combinations \
		-config_file=$$(ls experiments/sweeps/sweep_fc_*.yaml | sort -V | tail -n 1)

num_combinations_cnn:
	python cli.py num-combinations \
		-config_file=$$(ls experiments/sweeps/sweep_cnn_*.yaml | sort -V | tail -n 1)

# pytorch_fully:
# 	jupyter nbconvert --execute --to html source/notebooks/pytorch_fully_connected.ipynb
# 	mv source/notebooks/pytorch_fully_connected.html output/pytorch_fully_connected.html

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: data tests remove_logs pytorch_fully

## Delete all generated files (e.g. virtual)
clean:
	rm -f data/raw/*.pkl
	rm -f data/raw/*.csv
	rm -f data/processed/*
