# Checkpoint manifest

`write_checkpoint_manifest.py` writes the machine-readable manifest under the
contract scratch root after every save. It records exposure rows/prompts,
trajectories, optimizer step, tokens consumed where available, code and data
revisions, split hash, model/tokenizer revision, checkpoint kind, file list,
sizes, SHA-256 hashes, and load instructions.

SFT durable weights correspond to 25K through 200K in 25K increments. OPD
durable weights correspond to 25%, 50%, 75%, and 100%. At any moment only the
latest completed cadence has full training state; older saves are weights-only.
At completion the final SFT and OPD checkpoints retain full state.
