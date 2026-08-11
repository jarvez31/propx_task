.PHONY: setup run test lint notebook clean

setup:                ## install runtime + dev dependencies
	pip install -r requirements-dev.txt

run:                  ## extract attributes for the default area -> outputs/roof_attributes.json
	python -m roofkit --location karlsplatz

test:                 ## run the unit tests
	pytest -q

lint:                 ## static checks
	ruff check roofkit tests

notebook:             ## open the interactive playground
	jupyter lab roof_attributes.ipynb

clean:                ## remove caches (keeps the DEM tile cache)
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache
