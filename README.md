# Fast-OSVOS (PyTorch)

## Requirements
This project uses pipenv, which is the officially
recommended Python packaging tool from Python.org

```
pipenv --python 3.5
pipenv shell
pipenv run pip install --no-deps torchvision
pipenv run pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pipenv install

cp src/config/mypath.py.example src/config/mypath.py
vi src/config/mypath.py
```

The last command has to be run because torchvision
depends on torch, which does not provide a pip package.
Therefore the installation fails, although technically
the dependency is already installed. pipenv unfortunately
does support skipping dependencies for now.

