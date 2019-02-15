install:
	pipenv install

dev-install:
	pipenv install --dev

yapf:
	pipenv run isort -y deepquantiles/
	pipenv run yapf -vv -ir deepquantiles/
	pipenv run isort -y tests/
	pipenv run yapf -vv -ir tests/

lint:
	pipenv run flake8 .
	pipenv run pydocstyle .
	pipenv run mypy .

clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

test:
	pipenv run pytest --cov=deepquantiles

test-cov:
	pipenv run pytest --cov=deepquantiles --cov-report html --cov-report term

release:
	python setup.py sdist bdist_wheel
	twine upload dist/*
