# Processed Data Layout

Store local experiment-ready CSVs here.

- `standardized_examples.csv`
  Columns: `example_id,task_id,task_type,prompt,target,split`
- `client_assignments_low.csv`
  Columns: `example_id,client_id,heterogeneity_level`
- `client_assignments_high.csv`
  Columns: `example_id,client_id,heterogeneity_level`

The training smoke-test script joins these files on `example_id` and then tokenizes them
with the current Qwen tokenizer.
