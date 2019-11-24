install:
	# Update package dependences
	conda list -e > requirements.txt
	python setup.py install

remove:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf data/tmp/
	# Remove all instances of __pycache__
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf

rebuild: 
	@make remove
	@make install

run: 
	@make install
	pytest -vv
	@make remove
