#!/bin/bash

# table 1
echo "table 1"
python scripts/experiments_wacv23/tex/generate_table_sncalib-center.py
# table 2, 3
echo "table 2,3"
python scripts/experiments_wacv23/tex/generate_table_wc14-center.py
# table appendix: lens distortion
echo "table appendix"
python scripts/experiments_wacv23/tex/generate_table_lens_distortion.py
# figure 2: segment reprojection loss
echo "figure 2"
python scripts/experiments_wacv23/figures/visualize_ndc_losses_multiple_datasets.py
# figure 3: sn-calib-test (main left, center, right)
echo "figure 3"
python scripts/experiments_wacv23/figures/summarize_results_sncalib-test-all.py
# evaluate projection error
echo "evaluate projection error"
python -m scripts.experiments_wacv23.tex.prepare_iou_results