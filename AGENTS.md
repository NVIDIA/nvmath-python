# Key Takeaways for nvmath-python Development

This file captures important lessons and context gathered while working on the `nvmath-python` project, covering documentation, CI/CD, environments, and development patterns.

## 1. Environment and Tooling
* **Virtual Environment**: Always activate the correct virtual environment (e.g., `source .venv/bin/activate`) before running tools like `pre-commit` or building Sphinx documentation (`make html`). This ensures that required dependencies (like `tomllib` in Python 3.11+) are available.
* **Pre-commit Hooks**: 
  * Running `pre-commit run -a` can download large environments (like `mypy` and `ruff`).
  * Because the default `~/.cache` directory may run out of space, **always** override the pre-commit cache location by setting `export PRE_COMMIT_HOME=/local/home/yhavrylko/.cache/pre-commit` before running hooks.
* **Artifactory Secrets**: Some tasks require Artifactory credentials (like updating locks). Load them by running `source ~/.zshrcd/secrets/artifactory.sh`.
* **UV Cache**: When running `uv` commands (like `inv lock`), use the custom cache directory by setting `export UV_CACHE_DIR=/local/home/yhavrylko/.cache/uv` to avoid filling up the default `~/.cache` directory.
* **Git Commit**: Be aware of custom git configurations, hooks, or aliases in the environment that might interfere with standard commands (e.g., an unknown `--trailer` option error).

## 2. Sphinx Documentation
* **Centralized Versions**: The versions for all MathDx components (cuBLASDx, cuFFTDx, cuSOLVERDx, etc.) have been centralized in a `mathdx_versions` dictionary inside `docs/sphinx/conf.py`. This acts as the single source of truth and powers both `rst_prolog` substitutions and `extlinks`.
* **Sphinx Roles**: 
  * Use custom `extlinks` like `:cublasdx_doc:\`api/other_tensors.html\`` for external C++ docs, rather than hardcoding raw URLs.
* **RST Formatting Strictness**: 
  * **Line Length**: The `ruff` linter enforces a strict line-length limit for docstrings (e.g., 92 characters). Long links using macros might exceed this and need re-wrapping.
  * **Indentation**: Explicit markup blocks (like `.. seealso::`) require proper blank lines and strict indentation for their content. Improper unindentation will cause Sphinx `docutils` warnings.

## 3. Python Development Patterns
* **Dynamic Property Docstrings**: When applying a custom decorator (like `docstring_decorator` in `nvmath.internal.utils`) to format docstrings dynamically with `{macro}` substitutions, you cannot directly assign to a property getter's `__doc__`. Instead, you must instantiate a new `property` object with the formatted docstring and the original `fget`, `fset`, and `fdel` methods, then set it back on the class using `setattr`.

## 4. Relevant File Structure
* **`nvmath/device/`**: Contains the core Python implementation for device APIs.
  * `cublasdx.py`: Implementation of `Matmul` and cuBLASDx traits.
  * `cufftdx.py`: Implementation of `FFT` and cuFFTDx traits.
  * `cusolverdx.py`: Implementations for various cuSOLVERDx solvers.
  * `common.py`: Shared utilities and tensor handling.
* **`docs/sphinx/`**: Contains the Sphinx documentation configuration and source files.
  * `conf.py`: Configuration file where `mathdx_versions`, `rst_prolog` substitutions, and `extlinks` are defined.
  * `device-apis/`: ReStructuredText files documenting the device APIs.
    * `cublas.rst`: cuBLASDx documentation, including the feature readiness table.
    * `cufft.rst`: cuFFTDx documentation.
    * `cusolver.rst`: cuSOLVERDx documentation.
* **`nvmath/internal/utils.py`**: Contains internal utilities, including the `docstring_decorator` used to dynamically format property docstrings.
* **CI and Dependencies (`.ci/` and `pyproject.toml`)**:
  * `pyproject.toml`: Defines optional dependencies and test groups (e.g. `tests-dx-dev-cu13`).
  * `.ci/mr-tests.json` & `.ci/nightly-tests.json`: Define the test environments triggered during CI runs (e.g. `test-cu13-dx-dev`).
  * `.ci/update_locks.py`: Python script used to map dependency sets to `pyproject.toml` groups and generate lock files using `uv`.
  * `.ci/locks/`: Contains the generated dependency lock files (e.g. `pylock.test-cu13-dx-dev.toml`).
  * `.ci/dev/script/`: Contains bash scripts (e.g. `install-libmathdx-cu13-dev.sh`) to download and install library dependencies from Artifactory via tar archives.

## 5. CI / CD and Dependency Locks
* **Dependency Locks for CI**: CI environments manage dependencies strictly using locked files found in `.ci/locks/`. 
  * When adding or modifying an environment (like changing `test-cu13-dx` to `test-cu13-dx-dev` in `.ci/mr-tests.json`), the lock file must be regenerated.
  * To update dependency locks, you need `uv` installed, an active virtual environment, and credentials sourced. The typical flow is:
    ```bash
    export UV_CACHE_DIR=/local/home/yhavrylko/.cache/uv
    source .venv/bin/activate
    source ~/.zshrcd/secrets/artifactory.sh
    inv lock --deps test-cu13-dx-dev
    ```
  * **Keep Versions Aligned**: The version of `libmathdx` downloaded via tar in scripts like `.ci/dev/script/install-libmathdx-cu13-dev.sh` **must always be aligned** with the version required by the generated lock files (which are derived from the bounds set in `pyproject.toml`). If `pyproject.toml` or the lock dictates `libmathdx` dev version `490`, the `MR` variable inside the bash script must also be set to `490`.

## 6. GitLab Integration
* Currently, the local `user-MaaS_GitLab` MCP primarily supports read-only operations. To open an MR programmatically, use Git push options if the branch is already pushed:
  ```bash
  git commit --allow-empty -m "Trigger MR creation"
  git push -o merge_request.create -o merge_request.target=main origin <branch-name>
  ```
* **CodeRabbit Configuration**: The project uses CodeRabbit for automated code reviews. 
  * The configuration is stored in `.coderabbit.yaml` at the root of the repository.
  * To automatically trigger reviews on every Merge Request, ensure `auto_review.enabled` is set to `true` in the configuration.