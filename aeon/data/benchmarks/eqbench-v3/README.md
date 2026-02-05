Benchmark repo: https://github.com/EQ-bench/eqbench3

Sample command to run the benchmark:
(I modified it a bit to use ollama for the local model. You must also run `ollama serve` in another terminal before running this. You don't need to manually call `ollama run <model>` though. )
```
python eqbench3.py \
    --test-model olmo-3:7b-instruct \
    --judge-model gpt-5.2 \
    --no-elo \
    --iterations 1 \
    --threads 1
```

This creates a new entry in eqbench3_runs.json. I added a tiny script to extract the latest entry and write to a file in results/extracted which can then be copied to the aeon dir.

The following snippet shows loading one results file and extracting the actual generations:

```
with open(path, 'r') as f:
    raw = json.load(f)

# actual generations from the model being benchmarked
raw['60539694_olmo-3_7b-instruct']['scenario_tasks']['1']

# translate score to eqbench-v3 leaderboard scale (0-100)
raw['60539694_olmo-3_7b-instruct']['results']['average_rubric_score'] * 5
```
