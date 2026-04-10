# %%
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

task_names = [
    "task075_squad1.1_answer_generation",
    
	"task1291_multi_news_summarization",

    "task195_sentiment140_classification",	
]

# Canonical repo id is hyphenated; underscore form is fallback.
dataset_candidates = [
    "Muennighoff/natural-instructions",
]

last_error = None
full_dataset = None
for dataset_id in dataset_candidates:
    try:
        # Load default config and bypass strict split verification.
        full_dataset = load_dataset(
            dataset_id,
            split="train",
            verification_mode="no_checks",
        )
        break
    except (DatasetNotFoundError, ValueError) as err:
        last_error = err

if full_dataset is None:
    raise RuntimeError(
        f"Could not load any known dataset id: {dataset_candidates}. Last error: {last_error}"
    )

# Task ids are stored as data fields, not builder configs.
task_key_candidates = ["task_name", "task", "Task", "id", "task_id"]
task_key = next((k for k in task_key_candidates if k in full_dataset.column_names), None)

if task_key is None:
    raise RuntimeError(
        f"Could not find a task column. Available columns: {full_dataset.column_names}"
    )

raw_datasets = {}
for task in task_names:
    task_subset = full_dataset.filter(lambda ex: ex[task_key] == task)
    if len(task_subset) == 0 and task_key == "id":
        task_subset = full_dataset.filter(
            lambda ex: isinstance(ex["id"], str) and ex["id"].startswith(task)
        )
    raw_datasets[task] = task_subset

loaded_non_empty = sum(len(ds) > 0 for ds in raw_datasets.values())
print(f"Loaded {loaded_non_empty}/{len(task_names)} tasks with at least one example.")

# %%
# testing purposes

# from collections import Counter

# # Show which requested tasks are missing and suggest close matches.
# counts = Counter(full_dataset[task_key])

# print(f"task key in use: {task_key}")
# print("\nRequested task counts:")
# for t in task_names:
#     print(f"- {t}: {counts.get(t, 0)}")

# missing = [t for t in task_names if counts.get(t, 0) == 0]
# print(f"\nMissing tasks ({len(missing)}): {missing}")

# if missing:
#     all_tasks = list(counts.keys())
#     print("\nClosest available task ids:")
#     import difflib

#     for t in missing:
#         close = difflib.get_close_matches(t, all_tasks, n=5, cutoff=0.45)
#         print(f"\n{t}")
#         for c in close:
#             print(f"  -> {c} (count={counts[c]})")

# %%
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
CLIENTS_PER_GROUP = 5

TASK_GROUPS = {
    "qa": [
        "task075_squad1.1_answer_generation",
    ],
    "summarization": [
        "task1291_multi_news_summarization",
    ],
    "classification": [
        "task195_sentiment140_classification",
    ],
}


# %%
def high_hetero_split(tokenized_tasks, task_groups, clients_per_group=2):
    total_clients = clients_per_group * len(task_groups)
    print(f"Total clients: {total_clients} ({clients_per_group} per group × {len(task_groups)} groups)")
    
    client_data = []

    return client_data

# %%

def low_hetero_split(tokenized_tasks, num_clients):
    all_samples = []
    for task_name, ds in tokenized_tasks.items():
        for i in range(len(ds)):
            all_samples.append(ds[i])

    random.shuffle(all_samples)

    client_data = [[] for _ in range(num_clients)]
    for i, sample in enumerate(all_samples):
        client_data[i % num_clients].append(sample)

    print(f"[Low Hetero] {num_clients} clients, ~{len(client_data[0])} samples each")
    return client_data

# %%
def standardize(example, task_name):
    # Natural Instructions typically has 'input' and 'output' fields
    return {
        "prompt": example.get("input", example.get("Instance", {}).get("input", "")),
        "target": example.get("output", example.get("Instance", {}).get("output", [""])[0]),
        "task": task_name
    }

standardized = {}
for task, ds in raw_datasets.items():
    standardized[task] = ds.map(lambda x: standardize(x, task), remove_columns=ds.column_names)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
MAX_INPUT_LEN = 400
MAX_TARGET_LEN = 128

def tokenize(example):
    model_input = tokenizer(
        example["prompt"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        example["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    ).input_ids

    # Replace padding token id with -100 so loss ignores them
    labels = [-100 if t == tokenizer.pad_token_id else t for t in labels]
    model_input["labels"] = labels
    return model_input

tokenized = {}
for task, ds in standardized.items():
    tokenized[task] = ds.map(tokenize, remove_columns=["prompt", "target", "task"])

# %%
# High hetero: total = CLIENTS_PER_GROUP × 3 groups
high_clients = high_hetero_split(tokenized, TASK_GROUPS, 
                                  clients_per_group=CLIENTS_PER_GROUP)

# Low hetero: match the same total for fair comparison
NUM_CLIENTS = CLIENTS_PER_GROUP * len(TASK_GROUPS)
low_clients  = low_hetero_split(tokenized, num_clients=NUM_CLIENTS)

print(f"Total clients in both settings: {NUM_CLIENTS}")


