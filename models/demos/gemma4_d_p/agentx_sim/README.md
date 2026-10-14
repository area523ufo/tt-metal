# AgentX simulator (Gemma4 prefill)

Host only. Run from this directory.

```
# traces
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset --local-dir <dir>

# one config over the whole corpus
./agentx_sim.py --traces <dir>/traces.jsonl --config 8192x8192:fcfs --concurrency 64 --calibrated

# the tt-metal#60408 sweep
./sweep_60408.sh <dir>/traces.jsonl sweep.jsonl

# a slice for a silicon replay, its prediction, and the comparison with the measured replay
./agentx_slice.py build --traces <dir>/traces.jsonl --out slice.json --concurrency 48 --start 600 --window 120
./agentx_slice.py predict --slice slice.json --calibrated
./agentx_slice.py compare --slice slice.json --measured replay.json --calibrated
```

`--help` on each script lists its options.
