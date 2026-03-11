# evaluatorq-py Examples

This directory contains Python examples demonstrating the capabilities of the evaluatorq-py library.

## Directory Structure

```
examples/lib/
├── basics/                          # Core examples and entry points
│   ├── examples.py                  # Main entry point
│   ├── example_runners.py           # Simulated delay examples
│   ├── pass_fail_simple.py          # Simple pass/fail evaluation
│   ├── eval_reuse.py                # Reusable jobs and evaluators
│   ├── simple_local_eval.py         # Simple local evaluation
│   └── llm_eval_with_results.py     # LLM-based evaluation
├── datasets/                        # Dataset-based evaluations
│   ├── dataset_example.py           # Orq platform dataset evaluation
│   └── country_unit_test.py         # Unit test style with Orq dataset
├── structured/                      # Structured evaluation results
│   ├── structured_rubric_eval.py    # Multi-criteria quality rubric
│   ├── structured_sentiment_eval.py # Sentiment distribution
│   ├── structured_safety_eval.py    # Toxicity/safety scoring
│   └── path_organization.py        # Path-based dashboard organization
├── integrations/                    # Framework integrations
│   └── langchain/                   # LangChain / LangGraph
│       ├── langchain_integration_example.py   # Basic LangChain agent eval
│       ├── langgraph_integration_example.py   # Basic LangGraph agent eval
│       ├── langgraph_research_eval.py         # Research agent with dynamic instructions
│       └── langgraph_dataset_eval.py          # Dataset-based LangGraph eval
├── cli/                             # CLI integration examples
│   ├── example_using_cli.py
│   ├── example_using_cli_two.py
│   ├── example_llm.py
│   └── example_cosine_similarity.py
└── utils/                           # Shared evaluator utilities
    └── evals.py
```

## Running Examples

```bash
# Basics
python3 examples/lib/basics/examples.py
python3 examples/lib/basics/pass_fail_simple.py

# Datasets (requires ORQ_API_KEY)
ORQ_API_KEY=... python3 examples/lib/datasets/dataset_example.py

# Structured
python3 examples/lib/structured/structured_rubric_eval.py

# Integrations (requires OPENAI_API_KEY)
ORQ_API_KEY=... OPENAI_API_KEY=... python3 examples/lib/integrations/langchain/langgraph_research_eval.py
ORQ_API_KEY=... OPENAI_API_KEY=... DATASET_ID=... python3 examples/lib/integrations/langchain/langgraph_dataset_eval.py

# CLI
python3 examples/lib/cli/example_using_cli.py
```

## Prerequisites

```bash
pip install evaluatorq python-dotenv

# For LLM examples
pip install anthropic openai

# For LangGraph integration
pip install langchain-openai langgraph
```

Set API keys:
```bash
export ORQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```
