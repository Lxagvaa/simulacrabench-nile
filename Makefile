.PHONY: setup eval eval-fast eval-final test official zip report

setup:       ## environment, starter kit, sandbox data, TabPFN checkpoint
	./scripts/setup.sh
eval:        ## all three sandbox instruments in one process, development budget
	uv run python scripts/eval_main.py submission unhcr,world_bank,unicef
eval-fast:   ## UNICEF only
	uv run python scripts/local_eval.py --kit starter_kit --instruments unicef
eval-final:  ## final-phase conditions: TRAIN+DEV visible, TEST scored, 3300 s
	uv run python scripts/eval_main.py submission unhcr,world_bank,unicef 2
test:        ## adversarial frames: NaN in GIVEN, off-schema and numeric values
	uv run python scripts/robust_test.py
official:    ## the organisers' grader on all three sandboxes
	./scripts/official_check.sh
zip:         ## dist/submission.zip, validated
	./scripts/build_zip.sh
report:      ## technical report: report/main.tex -> report/main.pdf
	cd report && tectonic -X compile main.tex
