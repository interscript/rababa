LIB_LOCAL_PATH = "/home/jair/WORK/Farsi/rababa"

import sys

sys.path.append(LIB_LOCAL_PATH)


import lib.get_assets as assets
import lib.model0_9 as l0_9
import lib.rules_and_coord as m0_9


#print("Number of arguments:", len(sys.argv), "arguments.")
#print("Argument List:", str(sys.argv))

if len(sys.argv) != 2:
    print("API ran as follows:")
    print("python transliterate.py 'مزايایی'")
    exit()

text = sys.argv[1]
print(m0_9.run_transcription_0(text))
