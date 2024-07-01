# nvmath-python requirements
Dependencies are organized with requirements.txt files which can be use to set up virtualenvs with all required development tools to build docs, run tests, and build redistributable wheels.  Different requirements are necessary for installation with [pip](https://pip.pypa.io/en/stable/) vs [conda](https://docs.conda.io/en/latest/).

## Pip: Top-level package requirements files
Prefer using these `requirements/pip-<name>.txt` files for development in pip managed virtualenvs.  These include all relevant requirements sets and package extras.

### Pip: Supported configurations for CTK wheels

| requirements.txt | Extras | Python Support | Platform Support | CUDA | Purpose |
| ---------------- | ------ | ------- | ------- | ----- | ---- |
| `requirements/pip-dev-cu11.txt` | `cu11` | `3.9-3.12` | `linux_x86_64` | `11.x` | Development environment: `CUDA-11`  |
| `requirements/pip-dev-cu11-torch.txt` | `cu11`,`torch_cu11` | `3.9-3.11` | `linux_x86_64` | `11.8` | Development environment: `CUDA-11` + torch |
| `requirements/pip-dev-cu12-dx.txt` | `cu12` | `3.9-3.12` | `linux_x86_64`, `linux_aarch64` | `12.x` (latest) | Development environment: `CUDA-12` + DX APIs |
| `requirements/pip-dev-cu12-dx-torch.txt` | `cu12`, `dx`, `torch_cu12` | `3.9-3.11` | `linux_x86_64`, `linux_aarch64` | `12.1` | Development environment: `CUDA-12` + DX APIs + torch |

### Pip: Development usage
The requirements files provide dependencies only.  The nvmath-python package itself must also be installed, typically in editable mode for development.  Extras are not required to be specified on the editable install assuming the right requirements.txt has been installed in virtualenv.

*Note*: For testing wheel/RPATH support locally, currently it requires to build in the non-editable mode (no `-e` flag).

#### Install with pip
Typically this is done inside a [virtualenv](https://docs.python.org/3/library/venv.html).
```bash
$ pip install -r requirements/pip-dev-<name>.txt
$ pip install -e .
```

#### Install with pipenv
See [pipenv docs](https://pipenv.pypa.io/en/latest/) for reference

```bash
$ pipenv install -r requirements/pip-dev-<name>.txt
$ pipenv shell
(nvmath-python) $ pip install -e .
```

### Pip: Fine-grained requirements
Requirements for specific functionality are broken out into subsets.  These fine-grained requirements are included by the top-level requirements sets.

| requirements.txt | Functionality
| ---------------- | ------- |
| requirements/pip/build-wheel.txt | Utilities to build and validate wheels |
| requirements/pip/dev-tools.txt | Development utilities and tools |
| requirements/pip/docs.txt | Build documentation |
| requirements/pip/nvmath-python.txt | nvmath-python core requirements |
| requirements/pip/nvmath-python-cu11.txt | nvmath-python `[cu11]` extra requirements.  Support CUDA-11.x via wheels. |
| requirements/pip/nvmath-python-cu12.txt | nvmath-python `[cu12]` extra requirements.  Support CUDA-12.x via wheels. |
| requirements/pip/nvmath-python-dx.txt | nvmath-python `[dx]` extra requirements.  Enable device APIs. |
| requirements/pip/nvmath-python-sysctk11.txt | Runtime requirements to support nvmath-python with system installed CTK-11.x |
| requirements/pip/nvmath-python-sysctk12.txt | Runtime requirements to support nvmath-python with system installed CTK-12.x |
| requirements/pip/tests.txt | Test dependencies |
| requirements/pip/torch-cu11.txt | Enable torch use in tests and examples via wheels for CUDA-11.8 |
| requirements/pip/torch-cu12.txt | Enable torch use in tests and examples via wheels for CUDA-12.1 |
