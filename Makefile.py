all:
	$(MAKE) install && $(MAKE) test

flake8:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 `find . -name \*.py | grep -v setup.py | grep -v /doc/ | grep -v __init__.py`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
	python -m pytest --pyargs graph_energy --cov-report term-missing --cov=graph_energy 

install:
	$(MAKE) reqs
	python setup.py install --user || python setup.py install

reqs:
	pip install -r requirements.txt

hyak_install:
	pip install -r hyak/requirements.txt
	python setup.py install
