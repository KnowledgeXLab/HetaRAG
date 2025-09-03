# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Additional custom targets for HRAG documentation

# Install documentation dependencies
install-deps:
	pip install -r requirements.txt

# Clean all build artifacts
clean-all:
	rm -rf build/
	rm -rf source/_autosummary/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build documentation with auto-reload for development
dev:
	sphinx-autobuild source build/html --port 8000 --open-browser

# Build documentation with all warnings as errors
strict:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -W

# Build documentation with coverage report
coverage:
	$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Check for broken links
linkcheck:
	$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Build documentation for different formats
pdf:
	$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

epub:
	$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Generate API documentation
api:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -D autodoc_default_options.members=True

# Build documentation with internationalization
i18n:
	$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Build documentation for deployment
deploy:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Documentation built successfully!"
	@echo "You can find the built documentation in $(BUILDDIR)/html/"

# Quick build for testing
quick:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -q

# Build with parallel processing
parallel:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -j auto

# Validate documentation structure
validate:
	@echo "Validating documentation structure..."
	@python -c "import sphinx; print('Sphinx version:', sphinx.__version__)"
	@python -c "import sphinx_rtd_theme; print('RTD theme available')"
	@echo "Documentation structure validation completed."

# Setup development environment
setup-dev: install-deps validate
	@echo "Development environment setup completed."
	@echo "Run 'make dev' to start the development server."

# Full build with all checks
full-build: clean-all validate strict linkcheck coverage
	@echo "Full build completed with all checks passed!"

# Default target
all: html
