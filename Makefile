PY ?= python

.PHONY: verify figdata tables gates figures static clean-figdata

verify: figdata gates static
	@echo "=== all reproduction gates passed ==="

figdata:
	$(PY) paper/build_paper_numbers.py --bench nextgqa
	$(PY) paper/build_paper_numbers.py --bench qvhme

tables:
	$(PY) paper/gen_tables.py

gates: tables
	$(PY) analysis/c4_iou_high.py
	$(PY) analysis/c1_gc_sign.py
	$(PY) analysis/c2_sggt_ci.py
	$(PY) paper/gen_supp_numbers.py
	$(PY) analysis/a4_loo_reference.py

# Figures 4, 5, S1 and S2 -- the only images in the repository.
figures:
	$(PY) paper/make_fig4.py
	$(PY) paper/make_fig5.py
	$(PY) analysis/a2_threshold_sensitivity.py
	$(PY) analysis/a3_draw_variance.py

# Question-conditioned grounding check: the only body-level read of
# `predicted_answer` in any adapter must be the GVLLM_ANSCOND one.
static:
	$(PY) tools/check_no_answer_leak.py
