

import argparse
import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST
from brat_scoring.constants_sdoh import STATUS_TIME, TYPE_LIVING, STATUS_EMPLOY
from brat_scoring.scoring import score_brat_sdoh
from brat_scoring.constants_sdoh import LABELED_ARGUMENTS as SDOH_LABELED_ARGUMENTS


def get_argparser():
    parser = argparse.ArgumentParser(description = 'compares and scores two directories of brat files and summaries the performance in a csv file')
    parser.add_argument('gold_dir',        type=str, help="path to input directory with gold labels in BRAT format")
    parser.add_argument('predict_dir',     type=str, help="path to input directory with predicted labels in BRAT format")
    parser.add_argument('output_path',      type=str, help="path to output csv with scores")
    parser.add_argument('--labeled_args',  type=str, default=SDOH_LABELED_ARGUMENTS, nargs='+', help=f'span only arguments')
    parser.add_argument('--score_trig',    type=str, default=OVERLAP, help=f'equivalence criteria for triggers, {{{EXACT}, {OVERLAP}, {MIN_DIST}}}')
    parser.add_argument('--score_span',    type=str, default=EXACT,   help=f'equivalence criteria for span only arguments, {{{EXACT}, {OVERLAP}, {PARTIAL}}}')
    parser.add_argument('--score_labeled', type=str, default=LABEL,   help=f'equivalence criteria for labeled arguments, {{{EXACT}, {OVERLAP}, {LABEL}}}')
    parser.add_argument('--include_detailed', default=False, action='store_true',  help=f'generate document-level scores in addition to corpus-level scores')
    parser.add_argument('--loglevel',      type=str, default='info',  help='Provide logging level. Example --loglevel debug' )
    return parser

def main(args):
    '''
    This function scores a set of labels in BRAT format, relative to a set of
    gold labels also in BRAT format. The scores are saved in a comma separated
    values (CSV) file with the following columns:

    event - events type, like Alcohol, Drug, Employment, etc.
    argument - event trigger and argument, like Trigger, StatusTime, History, etc.
    subtype	- subtype labels for labeled arguments, like current or past for StatusTime
    NT - count of true (gold) labels
    NP - count of predicted labels
    TP - counted true positives
    P - precision
    R - recall
    F1 - f-1 score - harmonic mean of precision and recall

    Example (without commas for readability):
    event         argument             subtype             NT      NP      TP      P       R       F1
    Alcohol       StatusTime           current             63      47      23     0.49    0.37    0.42
    Alcohol       StatusTime           none                68      98      53     0.54    0.78    0.64
    Alcohol       StatusTime           past                31      3       1      0.33    0.03    0.06
    Alcohol       Trigger              N/A                162     148     121     0.82    0.75    0.78
    Alcohol       Type                 N/A                 32      2       1      0.50    0.03    0.06
    ...
    Employment    History              N/A                 17      0       0      0.00    0.00    0.00
    Employment    StatusEmploy         employed            27      13      11     0.85    0.41    0.55
    Employment    StatusEmploy         none                0       2       0      0.00    0.00    0.00
    Employment    StatusEmploy         on_disability       6       0       0      0.00    0.00    0.00
    Employment    StatusEmploy         retired             34      1       1      1.00    0.03    0.06
    Employment    StatusEmploy         student             1       0       0      0.00    0.00    0.00
    Employment    StatusEmploy         unemployed          22      29      10     0.34    0.45    0.39
    Employment    Trigger              N/A                 90      57      42     0.74    0.47    0.57
    Employment    Type                 N/A                286      36      13     0.36    0.05    0.08
    ...


    '''

    arg_dict = vars(args)
    logging.basicConfig(filename=arg_dict["output_path"]+'_out.log', filemode='w',encoding='utf-8', level=logging.DEBUG)


    score_brat_sdoh( \
        gold_dir = arg_dict['gold_dir'],
        predict_dir = arg_dict["predict_dir"],
        labeled_args = arg_dict["labeled_args"],
        score_trig = arg_dict["score_trig"],
        score_span = arg_dict["score_span"],
        score_labeled = arg_dict["score_labeled"],
        output_path = arg_dict["output_path"],
        include_detailed = arg_dict["include_detailed"],
        loglevel = arg_dict["loglevel"])



if __name__ == '__main__':

    parser = get_argparser()
    args = parser.parse_args()

    main(args)
