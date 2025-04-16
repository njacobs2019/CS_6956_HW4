**Creating the environment**
```
conda create --name env_name python=3.13
pip install -r requirements.txt
pre-commit install
```

**Running hooks on all code:**
```
pre-commit run --all-files
```

