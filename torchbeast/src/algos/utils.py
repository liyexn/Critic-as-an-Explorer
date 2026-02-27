# Original code Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the original license in the LICENSE file.
#
# Modified in 2025
#   - Added CAE, CAE+, CAE+ w/o U algorithms
#   - Simplified unrelated parts
#
# Additional modifications are released under CC BY-NC 4.0


import tensorflow as tf

def get_maxframes(uid, logger=None):
    files = tf.io.gfile.glob(f'{uid}/*.tfevents.*/')
    max_step = 0
    for file in files:
        try:
            for e in tf.compat.v1.train.summary_iterator(file):
                if e.step > max_step:
                    max_step = e.step
        except Exception as ex:
            if logger is not None:
                logger.info(f"ERROR: {ex}")
    return max_step
