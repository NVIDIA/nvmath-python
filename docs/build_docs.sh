#!/bin/bash

set -ex

# SPHINX_NVMATH_PYTHON_VER is used to create a subdir under _build/html
# (the docs/Makefile file for sphinx-build also honors it if defined)
if [[ -z "${SPHINX_NVMATH_PYTHON_VER}" ]]; then
    export SPHINX_NVMATH_PYTHON_VER=$(cat ../nvmath/_version.py | awk -F "=" '/__version__/ {gsub(/[^0-9.]/, "", $2); print $2}')
fi

# clean up
# TODO(leofang): the whole doc build is now blazingly fast that we do
# not care about the cached outputs. Revisit this when caching becomes
# neccessary.
rm -rf _build/
rm -rf _rtf/
rm -rf sphinx/_xml/
rm -rf sphinx/bindings/generated/
rm -rf sphinx/fft/generated
rm -rf sphinx/linalg/generated
rm -rf sphinx/device-apis/generated

# build the docs (in parallel)
SPHINXOPTS="-j 8" make html

# for debugging/developing (conf.py), please comment out the above line and
# use the line below instead, as we must build in serial to avoid getting
# obsecure Sphinx errors
#SPHINXOPTS="-v" make html

# to support version dropdown menu
cp sphinx/versions.json _build/html
cp sphinx/_templates/main.html _build/html/index.html

# ensure that the latest docs is the one we built
cp -r _build/html/${SPHINX_NVMATH_PYTHON_VER} _build/html/latest

# ensure that the Sphinx reference uses the latest docs
cp _build/html/latest/objects.inv _build/html
