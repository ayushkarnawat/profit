import os
import sys
import argparse
import multiprocessing as mp

from profit.dataset import converter

parser = argparse.ArgumentParser(prog=sys.argv[0], 
                                    description='Converts raw dataset to processed for modeling.',
                                    argument_default=argparse.SUPPRESS)
parser.add_argument('--raw_path', metavar='RAW DATASET PATH', type=str,
                    help='Relative path to the dataset', required=True, default='data/raw/vdgv570.csv')
parser.add_argument('--interim_path', metavar='INTERIM DATA PATH', type=str,
                    help='Relative path to save the interim dataset', default=None)
parser.add_argument('--processed_path', metavar='PROCESSED DATA PATH', type=str,
                    help='Relative path to save the final, processed dataset', default=None)
parser.add_argument('-x', '--x_name', metavar='DATA', type=str, help='Column name of dataset X', 
                    default='Variants')
parser.add_argument('-y', '--y_name', metavar='LABEL', type=str,
                    help='Column name of labels associated with the data X', default='Fitness')
parser.add_argument('-c', '--constraints', action='store_true',
                    help='Save molecules with constraints or not', default=False)
parser.add_argument('-n', '--n_workers', metavar='NUM WORKERS', type=int,
                    help='Number of workers', default=mp.cpu_count()-1)
args = vars(parser.parse_args())

# If the interim and processed filepaths are not provided, use default args
raw_fp = args['raw_path']
filename, ext = os.path.splitext(raw_fp)
split_file = filename.split('/')
if args['interim_path'] is None:
    args['interim_path'] = split_file[0] + '/interim/' + split_file[-1] + '_smiles' + ext
if args['processed_path'] is None:
    args['processed_path'] = split_file[0] + '/processed/' + split_file[-1] + '.sdf'

# X,y = load_csv(args['raw_path'], x_name=args['x_name'], y_name=args['y_name'], use_pd=True)
df = converter.convert_to_smiles(args['raw_path'], args['interim_path'], args['x_name'], 
                                 args['y_name'], save_constraints=args['constraints'])
converter.convert(args['interim_path'], args['processed_path'], x_name='SMILES', 
                  y_name=args['y_name'], n_workers=args['n_workers'])
    