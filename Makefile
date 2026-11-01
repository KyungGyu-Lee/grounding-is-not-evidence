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

figures:
	$(PY) paper/make_fig4.py
	$(PY) paper/make_fig5.py
	$(PY) analysis/c3_figures.py

# RELEASE_PLAN 6.5: grounding is question-conditioned. The only body-level read
# of `predicted_answer` in any adapter must be the GVLLM_ANSCOND one.
static:
	$(PY) tools/check_no_answer_leak.py
