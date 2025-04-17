pipeline:
	python scripts/pipeline/step1_convert_matlab_to_python.py
	python scripts/pipeline/step2_time_frequency_analysis.py
	python scripts/pipeline/step3_id_modulated_chans.py
	python scripts/pipeline/step4_spectral_parameterization.py
	python scripts/pipeline/step5_compute_intersection.py
	python scripts/pipeline/step6_aggregate_results.py

figures:
	python scripts/figures/figure_1.py
	python scripts/figures/figure_2.py
	python scripts/figures/figure_3.py
	python scripts/figures/figure_4.py
	python scripts/figures/figure_4-1.py
	python scripts/figures/figure_5.py
	python scripts/figures/figure_6.py
	python scripts/figures/figure_7.py
	python scripts/figures/figure_7-1.py
	python scripts/figures/figure_7-2ab.py
	python scripts/figures/figure_7-2cd.py

all: pipeline figures
