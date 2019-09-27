import os
import sys
import argparse
import multiprocessing as mp

from profit.dataset import converter

parser = argparse.ArgumentParser(prog=sys.argv[0], 
                                 description='Prepares raw dataset for modeling.',
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
parser.add_argument('-a', '--algo', metavar='ALGORITHM', type=str,
                    help="Algorithm to optimize molecule's 3D structure.", default='ETKDG')
args = vars(parser.parse_args())

# If the interim and processed filepaths are not provided, use default args
raw_fp = args['raw_path']
split_file = os.path.splitext(raw_fp)[0].split('/')
if args['interim_path'] is None:
    args['interim_path'] = "{0:s}/interim/{1:s}-constraints={2:b}.csv".format
        (split_file[0], split_file[-1], args['constraints'])
if args['processed_path'] is None:
    args['processed_path'] = "{0:s}/processed/{1:s}-constraints={2:b}-algo={3:s}.sdf".format\
        (split_file[0], split_file[-1], args['constraints'], args['algo'])

df = converter.convert_to_smiles(args['raw_path'], args['interim_path'], args['x_name'], 
                                 args['y_name'], save_constraints=args['constraints'])
converter.convert(args['interim_path'], args['processed_path'], x_name='SMILES', 
                  y_name=args['y_name'], algo=args['algo'], n_workers=args['n_workers'])
    