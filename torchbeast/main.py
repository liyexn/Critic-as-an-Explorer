# Original code Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the original license in the LICENSE file.
#
# Modified in 2025
#   - Added CAE, CAE+, CAE+ w/o U algorithms
#   - Simplified unrelated parts
#
# Additional modifications are released under CC BY-NC 4.0


from src.arguments import parser 

from src.algos.e3b import train as train_e3b
from src.algos.cae import train as train_cae
from src.algos.caep import train as train_caep
from src.algos.caep_wo_u import train as train_caep_wo_u


import os
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"


def main(flags):
    print(flags)
    flags.use_lstm = (flags.use_lstm == 1)


    if flags.num_contexts != -1:
        flags.fix_seed = True
        flags.env_seed = list(np.random.choice(range(100000), flags.num_contexts, replace=False))
        flags.env_seed = [int(i) for i in flags.env_seed]
        print(flags.env_seed)
    else:
        flags.fix_seed = False


    if flags.model == 'e3b':
        train_e3b(flags)
    elif flags.model == 'cae':
        train_cae(flags)
    elif flags.model == 'caep':
        train_caep(flags)
    elif flags.model == 'caep_wo_u':
        train_caep_wo_u(flags)
    else:
        raise NotImplementedError("model has not been implemented !!!")


if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags)




