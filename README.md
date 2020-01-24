# Exploring Evolutionary Protein Fitness


## Problem
Generate a list of top `k` protein evolutions that have a high likelihood of being stable. 
Additionally, find `k` additional compounds that, if given the fitness score, are most likely to improve the model.


## Notes
1. What happens if the optimal fitness sequence is not the the n-variant (aka 4-variant is all protein variants that have 4 mutations). For example, the optimal variant could only have 2 variants?? Additionally, knowing the distribution of the n=4 variants does not necessarily mean the k-variant protein is going to have the same distrubution. As a result, we might not actually even reach the most optimal variant.
2. Directed evolution, in-silico. Select a batch and train, then pick the highest variants (ones with the highest aquisition function), and train those in the next batch cycle, in a online fashion. This can help us both sample the whole search space as well as find the top-most variant in decreasing order of fitness.


## TODO
1. Implement graph-based models (in pytorch)
2. Ability to save trained models
    - at pre-defined timesteps/epochs
    - best model (based on the validation data)
    - within results/<dataset_name>/<model_label_num>/<time_step>/*.weights
    - save model configs: see how TAPE saves the info  
3. Load all filetype (from serializers.py) into pytorch/tfrecords dataset
4. Train using BO + batch_size=1 (or more)....batch determined by algorithm (see #2 in notes above)
5. Visualization
    - Plot results for meditation @39,40,41,54 for all 20 amino acids (w/ fitness score as heatmap), and label wildtype as black.