LIB_LOCAL_PATH = "/home/jair/WORK/Farsi/rababa"

import sys
import argparse
import tqdm

sys.path.append(LIB_LOCAL_PATH)


import lib.get_assets as assets
import lib.model0_9 as l0_9
import lib.rules_and_coord as m0_9


def transli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", dest="text", type=str, required=False)
    parser.add_argument("--text_file", dest="text_file", type=str, required=False)
    return parser


parser = transli_parser()
args = parser.parse_args()

if not args.text is None:
    text = args.text
    print(m0_9.run_transcription(text))
elif not args.text_file is None:
    text_file = args.text_file
    f = open(text_file, "r")
    for d in tqdm.tqdm(list(f.readlines())):
        try:
            print(m0_9.run_transcription(d))
        except:
            pass
else:
    print()
    raise TypeError("Arguments not supported")


