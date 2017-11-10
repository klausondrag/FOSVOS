# Fast-OSVOS (PyTorch)

## Requirements
This project uses pipenv, which is the officially
recommended Python packaging tool from Python.org

```
pipenv --python 3.6
pipenv shell
pipenv install
pipenv run pip install --no-deps torchvision==0.1.9
```

The last command has to be run because torchvision
depends on torch, which does not provide a pip package.
Therefore the installation fails, although technically
the dependency is already installed. pipenv unfortunately
does support skipping dependencies for now.

