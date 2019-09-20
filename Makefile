remove:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

install:
	python setup.py install

rebuild: 
	@make remove
	@make install

dataset:
	python profit/make_dataset.py --raw_path data/raw/vdgv570.csv -c 1 -n 4

run: 
	# Remove all instances of __pycache__ before running
	find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
	python -m test.test