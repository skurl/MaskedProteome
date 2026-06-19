#!/bin/bash

cd ~/masked_project

singularity exec --nv --bind $PWD:/app --pwd /app ./masked-proteome.sif python -u /app/bin/super_runner.py 2>&1 | tee run.log