   .PHONY: run_conda
   run_conda:
       @conda activate chorus-detection
       @python src/chorus_finder.py

   .PHONY: run_venv
   run_venv:
       @./venv/bin/python src/chorus_finder.py