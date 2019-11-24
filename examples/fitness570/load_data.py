import pandas as pd

from profit.dataset.preprocessors.transformer_preprocessor import TransformerPreprocessor
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.parsers.data_frame_parser import DataFrameParser
from profit.dataset.parsers.csv_parser import CSVFileParser

pp = TransformerPreprocessor(max_residues=10, out_size=10, use_pretrained=False)
mutator = PDBMutator(fmt='primary', remove_tmp_file=True)

df = pd.read_csv('data/raw/fitness570.csv')
parser = CSVFileParser(pp, mutator, 'Variants', 'Fitness')
out = parser.parse('data/raw/fitness570.csv')

print(out["dataset"][1][:10])
print(out["is_successful"])

# Parser reads either a CSV file (most common) with information about the 
# mutations and the fitness score. This requires a preproessor instance to 
# decide how to parse the data (which depends on the PDB mutator instance) to 
# determien whether we want a primary or tertiary structure. But how do we 
# put in the PDB ID/code? Should we set it using a method? The modified string 
# can just be the replace_with dict at the predefined locations. 
# 
# For example, do we pass in the 'TNTY' as a string, and set the positions of 
# the mutations as a seperate param. Then, we have to ensure that the positions 
# and length of the string are equal. 

# mutator = PDBMutator(fmt="primary")
# seq = 'TNTY'
# residues_to_modify = [39, 40, 41, 54]
# assert len(seq) == len(residues_to_modify)
# replace_with = {resid: seq[i] for i, resid in enumerate(residues_to_modify)}
# modified_seq = mutator.modify_residues('3gb1', replace_with=replace_with)
# transformer_pp = TransformerPreprocessor(out_size=15)
# embedding = transformer_pp.get_input_feats(modified_seq)
# print(embedding)