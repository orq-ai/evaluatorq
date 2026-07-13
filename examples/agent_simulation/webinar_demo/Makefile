# Agent Simulation webinar demo — provision the Bank of Holland agent and drive simulations.
# Requires ORQ_API_KEY in the environment or agent_build/.env.

AGENT ?= boh_creditcard_agent_demo
PERSONAS ?= 3
SCENARIOS ?= 3
DATAPOINTS ?= boh_datapoints.jsonl

.PHONY: provision generate simulate rerun upload help

help:
	@echo "make provision  - create Bank of Holland agent + tools + KB in Orq (idempotent)"
	@echo "make generate   - generate personas x scenarios -> $(DATAPOINTS)"
	@echo "make simulate   - run simulations against the agent, save datapoints"
	@echo "make rerun      - re-run the saved datapoints (compare after a change)"
	@echo "make upload     - push the datapoints to an Orq dataset"

provision:
	cd agent_build && uv run python provision.py

generate:
	uv run evaluatorq sim generate --target agent:$(AGENT) \
		--num-personas $(PERSONAS) --num-scenarios $(SCENARIOS) -o $(DATAPOINTS)

simulate:
	uv run evaluatorq sim run --target agent:$(AGENT) \
		--num-personas $(PERSONAS) --num-scenarios $(SCENARIOS) \
		--save-datapoints $(DATAPOINTS) --name boh

rerun:
	uv run evaluatorq sim simulate --target agent:$(AGENT) --datapoints $(DATAPOINTS) --name boh-rerun

upload:
	uv run evaluatorq sim upload-dataset -i $(DATAPOINTS) -n "Bank of Holland Sim Set"
