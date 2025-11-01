Official implementation for the paper "Co3: Bathymetric Data Completion with Context and Surface Consistency for Efficient Underwater Mapping".

* Important Note:

The main branch is currently under refactoring:
  -data_management: files for dataset generation and data processing;
  -losses: definitions of losses while training;
  -models: the models of completion network, context discriminator and surface discriminator;
  -trains: files for network training under various missing region scales;
  -test_dataset: dataset samples for training and inference;
  -samples: samples of the output data, incuding global and local scales.
  
The most complete code is in the ganbest branch, but not well-organized.

* Environment
  - Python 3.10.13
  - CUDA 12.4

