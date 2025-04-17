# LLM Evaluation Experiments

This directory contains scripts and data for evaluating LLM model performance. The current data is based on `gpt-4o-mini`. To evaluate other models, simply specify the desired model when invoking `llm_graph`.

## How to Run the Evaluation

### 1. (Optional) Generate Manual Workflow Output
Run `run_manual_workflow.py` to generate `manual_workflow.json`, which captures the results of a manually executed workflow.  
> **Note:** A sample `manual_workflow.json` is already included in this folder.

### 2. Run the LLM Workflow
Execute `run_llm_workflow.py` to generate a timestamped output file `llm_workflow_TIMESTAMP.json`. This file includes the LLM-generated workflow and associated metadata such as model name, timestamp, and system prompt.

### 3. Run Evaluation
Use `eval.py` to compare the LLM-generated results with the manual results and compute evaluation statistics:

```bash
python eval.py --llm_workflow name_of_llm_workflow.json
```

#### For Vibrational Tasks:
Add the `--vib_task` flag:

```bash
python eval.py --llm_workflow name_of_llm_workflow.json --vib_task
```
#### For Save File Tasks:
Add the `--save_task` flag:

```bash
python eval.py --llm_workflow name_of_llm_workflow.json --save_task
```
#### For Reaction Tasks:
Add the `--reaction_task` flag:

```bash
python eval.py --llm_workflow name_of_llm_workflow.json --reaction_task
```