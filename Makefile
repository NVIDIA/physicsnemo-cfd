install:
	pip install --upgrade pip && \
		pip install -e .

editable-install:
	pip install --upgrade pip && \
		pip install -e .[dev] --config-settings editable_mode=strict

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	pre-commit run interrogate -a

lint:
	pre-commit run markdownlint -a && \
	pre-commit run check-added-large-files -a

license: 
	python test/ci_tests/header_check.py --all-files

doctest:
	echo "Not implemented"

pytest: 
	python -m pytest test/ci_tests/test_evaluation.py -v --tb=short

coverage:
	echo "Not implemented"

docs:
	uv pip install -e ".[docs]"
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

all-ci: setup-ci black interrogate lint license install pytest doctest coverage
