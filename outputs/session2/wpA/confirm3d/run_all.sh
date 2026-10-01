#!/bin/bash
cd /Users/aaronramos/Desktop/SWIRL
printf '%s\n' confirm3d_primary confirm3d_twin | xargs -P 2 -n 1 outputs/session2/wpA/confirm3d/run_confirm3d.sh
