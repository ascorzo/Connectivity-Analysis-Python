# Setup

## Create separate environment (optional)
to avoid library conflicts with other projects. The environment named "andrea_connectivity" will only have the libraries installed for this project.

```shell
pip install virtual env
virtualenv andrea_connectivity
```

To use the environment you have to activate it. On a shell execute the path to this file

```
...\andrea_connectivity\Scripts\activate
```

This will put the current shell in the context of the environment. This means that the things that you install using pip will be installed in the environment.

## Installed required libraries

```shell
pip install -r requirements.txt
```

## Run

```shell
python Connectivity _calculation.py
```

