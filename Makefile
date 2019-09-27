remove:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	# Remove all instances of __pycache__
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf

install:
	python setup.py install

rebuild: 
	@make remove
	@make install

dataset:
	# Make dataset, allowing for constraints (`-c` command) within molecules
	python scripts/make_dataset.py --raw_path data/raw/vdgv570.csv -c -n 2 -a ETKDG

run: 
	# Remove all instances of __pycache__ before running
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf
	pytest -v