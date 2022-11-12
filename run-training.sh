#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "/home/students/wciezobka/agh/TrainingGAN/"
python3 test_run.py
