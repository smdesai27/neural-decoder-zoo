.PHONY: setup setup-torch test demo robustness lint clean

setup:            ## install core deps + package (editable)
	pip install -e ".[dev]"

setup-torch:      ## install with the full model zoo
	pip install -e ".[dev,torch]"

test:             ## run the test suite
	pytest -q

demo:             ## run the zoo on the synthetic dataset
	python -m ndz.run --dataset synthetic

robustness:       ## run the channel-dropout + signal-validity demo
	python -m ndz.run --dataset synthetic --robustness

clean:
	rm -rf results .pytest_cache **/__pycache__ src/*.egg-info
