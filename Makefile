.PHONY: create_venv
create_venv:
	python -m venv venv

.PHONY: run_venv
run_venv:
	venv\Scripts\python.exe src/chorus_finder.py $(URL)

.PHONY: create_conda_env
create_conda_env:
	conda env create -f environment.yml

.PHONY: run_conda
run_conda:
	@echo "Please activate the conda environment manually using 'conda activate chorus-detection' and then run 'python src/chorus_finder.py'"