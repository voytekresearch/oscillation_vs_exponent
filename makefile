pipeline:
	python scripts/pipeline/1_convert_matlab_to_python.py
	python scripts/pipeline/2_time_frequency_analysis.py
	python scripts/pipeline/3_id_modulated_chans.py
	python scripts/pipeline/4_spectral_parameterization.py
	python scripts/pipeline/5_compute_intersection.py
	python scripts/pipeline/6_aggregate_results.py
	python scripts/pipeline/7_permutation_stats.py

figures:
	python scripts/figures/figure_1.py
	python scripts/figures/figure_2.py
	python scripts/figures/figure_3ab.py
	python scripts/figures/figure_3cdef.py
	python scripts/figures/figure_4.py
	python scripts/figures/figure_5.py
	python scripts/figures/figure_6.py
	python scripts/figures/figure_7.py

all: pipeline figures
