# Exploring Evolutionary Protein Fitness


## Problem
Generate a list of top `k` protein evolutions that have a high likelihood of being stable. 
Additionally, find `k` additional compounds that, if given the fitness score,
are most likely to improve the model.


## Misc
To suppress the annoying pylint no-member warnings, add the following to `.vscode/settings.json` (if using vscode):
```json
"python.linting.pylintArgs": ["--generated-members=torch.*"]
```
