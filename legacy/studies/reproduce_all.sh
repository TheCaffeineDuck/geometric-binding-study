#!/bin/bash
# Reproduce all results from:
# "Geometric Selection Rules for Multi-Oscillon Cluster Stability"
# Estimated total: ~26 hours on Mac Mini (4 P-cores)
set -e
echo "=== Reproducing paper results ==="
echo "Started: $(date)"

echo "--- Step 1/16: Isolated baseline (Sec 2.3) ---"
python studies/01_single_reference.py

echo "--- Step 2/16: Pairwise cosine sweep (Table 1a) ---"
python studies/10_continuous_phase.py

echo "--- Step 3/16: Distance dependence sweep (Table 1b) ---"
python studies/distance_sweep.py

echo "--- Step 4/16: Cube binding (Table 2) ---"
python studies/02a_cube_phase2.py
python studies/02c_cube_adjacent_flip.py
python studies/cube_polarized_T1_CE12.py

echo "--- Step 5/16: Icosahedron binding (Table 2) ---"
python studies/02b_icosahedron_phase2.py

echo "--- Step 6/16: Tetrahedron + Octahedron (Table 2) ---"
python studies/04_new_geometries.py

echo "--- Step 7/16: Cross-edge universality (Table 2) ---"
python studies/03_cross_edge_universality.py

echo "--- Step 8/16: Ico CE=15 variants (Table 4) ---"
python studies/07b_ico_ce15_variants.py

echo "--- Step 9/16: Second potential (Table 3) ---"
python studies/09_second_potential.py

echo "--- Step 10/16: N=128 resolution (Sec 8.1) ---"
python studies/08_convergence_N128.py
python studies/09_selection_rule_N128.py

echo "--- Step 11/16: Spacing + long evolution (Sec 8.2-8.3) ---"
python studies/phase7bc_edge_length_and_extended.py

echo "--- Step 12/16: Centroid tracking (Sec 8.3) ---"
python studies/centroid_tracking.py

echo "--- Step 13/16: Self-organization ensemble (Table 5) ---"
echo "(This step takes ~14 hours)"
python studies/phase12_self_organization.py

echo "--- Step 14/16: Boundary condition test (Sec 8.6) ---"
python studies/boundary_test.py

echo "--- Step 15/16: Dissipation test (Sec 8.7) ---"
python studies/dissipation_test.py

echo "--- Step 16/16: Threshold robustness test (Sec 8.6-8.7) ---"
python studies/threshold_robustness.py

echo "=== Complete ==="
echo "Finished: $(date)"
