install:
	# Update package dependences
	conda list -e > requirements.txt
	python setup.py install

remove:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	# Remove all instances of __pycache__
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf

rebuild: 
	@make remove
	@make install

dataset:
	# Make dataset, allowing for constraints (`-c` command) within molecules
	python scripts/make_dataset.py --raw_path data/raw/fitness570.csv -c -n 2 -a ETKDG

run: 
	@make install
	pytest -vv
	@make remove
