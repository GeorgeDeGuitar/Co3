Official implementation for the paper "Co3: Bathymetric Data Completion with Context and Surface Consistency for Efficient Underwater Mapping".

(1) Important Note:

The main branch is currently under refactoring:
  -data_manage: files for dataset generation and data processing;
  -losses: definitions of losses while training;
  -models: the models of completion network, context discriminator and surface discriminator;
  -trains: files for network training under various missing region scales;
  
The most complete code is in the ganbest branch, but not well-organized.


(2) Quick Start: git clone https://github.com/GeorgeDeGuitar/Co3.git cd Co3 git checkout ganbest
