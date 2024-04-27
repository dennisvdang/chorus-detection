# Makefile

# Setup the Python environment and install dependencies
setup:
	python -m venv venv
	./venv/Scripts/activate
	pip install -r requirements.txt

# Run the chorus_finder.py script
run:
	python src/chorus_finder.py --url $(URL)

# Clean up generated files or directories
clean:
	rm -rf venv
	rm -rf __pycache__

# Example of how to make the environment setup and run commands dependent on each other
all: setup run