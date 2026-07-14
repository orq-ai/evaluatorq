# Agent Simulation webinar demo — provision the Bank of Holland agent and drive simulations.
# Requires ORQ_API_KEY in the environment or agent_build/.env.

AGENT ?= boh_creditcard_agent_demo
PERSONAS ?= 3
SCENARIOS ?= 3
DATAPOINTS ?= boh_datapoints.jsonl
MAX_TURNS ?= 10
PARALLELISM ?= 5

.PHONY: provision generate simulate dashboard rerun upload help

help:
	@echo "make provision  - create Sterling (Bank of Holland agent) + tools + KB in Orq (idempotent)"
	@echo "make generate   - generate personas x scenarios -> $(DATAPOINTS)"
	@echo "make simulate   - run the generated datapoints against the agent"
	@echo "make dashboard  - open the dashboard on this run store (.evaluatorq/sim-runs/)"
	@echo "make rerun      - re-run the saved datapoints (compare after a change)"
	@echo "make upload     - push the datapoints to an Orq dataset"

provision:
	cd agent_build && uv run python provision.py

generate:
	uv run evaluatorq sim generate --target agent:$(AGENT) \
		--num-personas $(PERSONAS) --num-scenarios $(SCENARIOS) -o $(DATAPOINTS)

simulate:
	uv run evaluatorq sim simulate --target agent:$(AGENT) \
		--input $(DATAPOINTS) --max-turns $(MAX_TURNS) \
		--parallelism $(PARALLELISM) --name sterling

dashboard:
	uv run evaluatorq dashboard .evaluatorq/sim-runs

rerun:
	uv run evaluatorq sim simulate --target agent:$(AGENT) --input $(DATAPOINTS) \
		--max-turns $(MAX_TURNS) --parallelism $(PARALLELISM) --name sterling-rerun

upload:
	uv run evaluatorq sim upload-dataset -i $(DATAPOINTS) -n "Bank of Holland Sim Set"
