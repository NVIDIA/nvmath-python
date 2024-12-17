# Version switcher

Version switcher is controlled by `docs/sphinx/_static/switcher.json` file.
When releasing a new version, remember to add it there.

The path to the version switcher in `conf.py` is relative to the webserver
root, so it should work both locally with `inv docs-view` (see how `tasks.py`
are tweaked to serve from `/cuda/nvmath-python`) and on `docs.nvidia.com`
(our docs are at `/cuda/nvmath-python` there as well).
