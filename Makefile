# target [target2 ...]: [prerequisites ...]
	# [command1
	#  command2
	# ........]

# Phony targets are not files
.PHONY: \
	build-local \
	build-global \
	upload \
	install-local \
	run-tensor-example \
	run-canny-pipeline

help:
	# tensorx-python
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... build-local"
	@echo "... build-global"
	@echo "... upload"
	@echo "... install-local"
	@echo "... run-tensor-example"
	@echo "... run-canny-pipeline"

build-local:
	# . .venv/bin/activate
	rm -rf dist
	python -m build

build-global:
	rm -rf dist
	cibuildwheel --output-dir dist

upload: build-global
	python -m twine upload dist/*

install-local: build-local
	# . .venv/bin/activate
	python -m pip uninstall tensorx-python
	python -m pip install --force-reinstall dist/*.whl

run-tensor-example:
	# . .venv/bin/activate
	python tensor_example.py

run-canny-pipeline:
	# . .venv/bin/activate
	python canny_pipeline.py
