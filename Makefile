.PHONY: corpus test sweep experiments report lint

corpus:        ## fetch primary legal sources and write the provenance manifest
	python -m lawful.corpus.build

test:          ## quote verification, legal unit tests, estimator tests
	pytest -q

lint:
	ruff check --select F,E9,B --ignore B905,B008 lawful experiments tests

sweep:         ## TEST: run open-weight agents through generated scenarios
	./experiments/run_agent_eval.sh

experiments:   ## REASON, TRAIN, JUDGE, PROBE, CONTROL, COLLUDE
	python experiments/02_reasoning.py --models qwen2.5-1.5b qwen3-4b --per-cell 12
	python experiments/04_train_grpo.py --steps 40
	python experiments/03_court_labels.py --llm qwen2.5-1.5b
	python experiments/05_case2env.py --agents qwen2.5-0.5b gemma3-1b qwen2.5-1.5b llama3.2-3b qwen3-4b
	python experiments/06_judges.py --pool 200 --probe-n 40
	python experiments/07_probes.py
	python experiments/08_control.py
	python experiments/09_collusion.py

report:        ## aggregate results and build report/index.html
	python experiments/analyze.py
	python experiments/build_report.py
