# Documentation

| File | Contents |
|---|---|
| `project_proposal.pdf` | the original project plan (phases 0–IV) |
| `concept_bank.md` | how the concept vocabulary was built, audited and cleaned, and its known problems |
| `upstream_molf_license.txt` | license of the upstream MoLF code (CC BY-NC 4.0) |
| `audit/` | the independent audit run on the final model before the results were frozen |

## audit/

The audit re-checked the executed pipeline end to end: checkpoint and config
hashes, split and gene-list integrity, the CONCH inputs of all 504 slides,
checkpoint selection, and the final test metrics. It passed with the caveats
listed in [`../results/README.md`](../results/README.md). The file names are
the ones the audit wrote.

| File | Contents |
|---|---|
| `FINAL_RUN2_PREZIP_FORENSIC_AUDIT.txt` | audit report |
| `FINAL_RUN2_PREZIP_FORENSIC_AUDIT.json` | the same report in machine-readable form |
| `FINAL_RUN2_PREZIP_AUDIT_console_output.txt` | full console output of the audit run |
| `FINAL_RUN2_PREZIP_AUDIT_SHA256SUMS.txt` | hashes of the two report files |
| `BUNDLE_SHA256SUMS.txt` | hashes of every file in the frozen result bundle, including the source code in `conch_molf/` and `experiments/` |
