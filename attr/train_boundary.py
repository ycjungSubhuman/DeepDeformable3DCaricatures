# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os
import sys
sys.path.append( os.path.dirname( os.path.abspath(__file__) ) )

import os.path
import numpy as np

from helper.logger import setup_logger
from helper.manipulator import train_boundary

#----------------------------------------------------------------------------

def _train_boundary(OUT_DIR, latent_codes, attr_scores, split_ratio=0.7, invalid_value=None):
  """Main function."""
  logger = setup_logger(OUT_DIR, logger_name='generate_data')


  boundary = train_boundary(latent_codes=latent_codes,
                            scores=attr_scores,
                            split_ratio=split_ratio,
                            invalid_value=invalid_value,
                            logger=logger)

  np.save(os.path.join(OUT_DIR, 'boundary.npy'), boundary)

  return boundary

#----------------------------------------------------------------------------
