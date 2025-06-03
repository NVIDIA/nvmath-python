# Minimal makefile for Sphinx documentation
SHELL=/bin/bash

# You can set these variables from the command line or environment
SPHINX_NVMATH_PYTHON_VER ?= $(shell [[ $$(< ../pyproject.toml) =~ [^a-zA-Z_]version\ =\ [^0-9.]*([0-9.]*) ]] && echo $${BASH_REMATCH[1]})
SPHINXOPTS    ?= -j auto -W
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = sphinx
BUILDDIR      ?= _build/html

help:
	@$(SPHINXBUILD) -b help "$(SOURCEDIR)" "$(BUILDDIR)/${SPHINX_NVMATH_PYTHON_VER}" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)
	rm -rf ${SOURCEDIR}/_xml/
	rm -rf ${SOURCEDIR}/bindings/generated/
	rm -rf ${SOURCEDIR}/fft/generated
	rm -rf ${SOURCEDIR}/linalg/generated
	rm -rf ${SOURCEDIR}/device-apis/generated

html: Makefile
	@echo BUILDDIR=${BUILDDIR}
	mkdir -p ${BUILDDIR}/latest
	$(SPHINXBUILD) -b $@ "$(SOURCEDIR)" "$(BUILDDIR)/${SPHINX_NVMATH_PYTHON_VER}" $(SPHINXOPTS) $(O)
	cp ${SOURCEDIR}/_templates/main.html $(BUILDDIR)/index.html
	cd $(BUILDDIR) && cp -a ${SPHINX_NVMATH_PYTHON_VER}/* latest/
	cp $(BUILDDIR)/latest/objects.inv $(BUILDDIR)


linkcheck: Makefile
	$(eval $@_BUILDDIR := $(shell mktemp -d))
	$(SPHINXBUILD) -b $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf $($@_BUILDDIR)

.PHONY: help Makefile clean html linkcheck
