# Exploring Evolutionary Protein Fitness Landscapes
Evaluting a protein's fitness function for all its evolutionary descendents (aka
variants) leads to extremely rugged, non-convex landscapes. Efficently
nativating this landscape is a hard optimization task, increasing in complexity
as the size of the protein increases. This makes it extremely hard to efficently
find the optimal variants. In this work, primarily based on [1], we explore a
protein's evolutionary landscape to efficently find a set of variants for a
desired design task.


## Setup
To install,
- Clone the github repo to a directory (i.e. `path/to/cloned/repo/`)
- `cd` into the cloned repo
- `python setup.py [install|develop]`


## Usage 

### Preprocessing
We have designed the preprocessing module to be easy to use for protein-related
sequences/tasks. To preprocess a sequence, use the following

```python
from profit.dataset.preprocessors import LSTMPreprocessor

template = list("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE")
preprocessor = LSTMPreprocessor("aa20")
encoded_seq = preprocessor.get_input_feats(template)
# array([12., 16., 18., 11., 10.,  9., 10.,  2.,  7., 11., 16., 10., 11.,
#         7.,  6., 16., 16., 16.,  6.,  0., 19.,  3.,  0.,  0., 16.,  0.,
#         6., 11., 19., 13., 11.,  5., 18.,  0.,  2.,  3.,  2.,  7., 19.,
#         3.,  7.,  6., 17., 16., 18.,  3.,  3.,  0., 16., 11., 16., 13.,
#        16., 19., 16.,  6.])
```

to obtain an sequence encoded representation of the protein. Similarly, for a
structural 3D representation of the protein,

```python
from profit.dataset.preprocessing import PDBMutator
from profit.dataset.preprocessors import EGCNPreprocessor

pos = [39, 40, 41, 54]
residues = ["T", "N", "T", "Y"]
mutator = PDBMutator(fmt="tertiary")
mol = mutator.mutate(pdbid="3gb1", replace_with=dict(zip(pos, residues)))

preprocessor = EGCNPreprocessor()
encoded = preprocessor.get_input_feats(mol)
print([arr.shape for arr in encoded])  # [(867, 63), (867, 867), (867, 867, 3)]
```

should be used. Extending this to use your own preprocessing pipeline, you have
to add a preprocessor class, depending on whether you want a sequence-based or
structure-based preprocessor.

```python

class MyFancyPreprocessor(SequencePreprocessor):

    def __init__(self, ...):
        # add variables/constants that do not change between each individual variant 
        pass

    def get_input_feats(self, seq):
        # preprocessing code here; return features as tuple
        pass
```

### Training
Building and training a model should be as simple as building any torch
module.

### Optimization
Given that the optimization is highly dependent on the model and the
preprocessed data, the optimization is the hardest to extend as there is no
underlying framework (currently) for how one should be structured. However, if
using the CbAS optimization technique, then one can simply call the procedure
with the required oracle, gpr, and generative model. See [examples](#examples)
for more details.


## Examples
The best way to understand how to best use the module is to refer to the
actual examples within the `examples/` folder.

### 3GB1 variants (regression task)
To demonstrate the capabilities of how the optimization procedure efficently
navigates a protein's fitness landscape, we train on actual protein variants,
provided by [2] (see `examples/gb1`). In particular, it is useful to take a look
at how the oracle (aka fitness evaluation function), gaussian process regressor
(aka the "quasi" ground truth evaluation function), and VAE (aka generative
model) are trained. More importantly, to understand how the CbAS algorithm uses
each of these "components" within its optimization scheme, take a look at
`cbas.py`.


## A note on performance
Although some portions of the code are built to support computations on the GPU
(i.e. model training code), a vast majority of the pre-processing steps occur on
the CPU, hidden under several abstraction layers. While these are cached after
the first run, they are severely unoptimized and quite slow. I would like to fix
this, it is not of high priority (for now).


## References
[1] Brookes, David H., Hahnbeom Park, and Jennifer Listgarten. "Conditioning by
adaptive sampling for robust design." arXiv preprint arXiv:1901.10060 (2019).

[2] Wu, Nicholas C., et al. "Adaptation in protein fitness landscapes is
facilitated by indirect paths." Elife 5 (2016): e16965.
