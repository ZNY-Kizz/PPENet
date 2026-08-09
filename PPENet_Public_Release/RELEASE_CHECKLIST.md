# Pre-upload checklist

- [ ] Obtain written disclosure approval for `data/EFKG-Public-Subset`.
- [ ] Replace `DATA_USE_NOTICE.md` with the provider-approved license.
- [ ] Create the GitHub repository, for example
      `https://github.com/<YOUR_GITHUB_ACCOUNT>/PPENet`.
- [ ] Replace the repository placeholder in the reviewer response letter.
- [ ] Confirm that no model weights, embeddings, candidate JSON, evidence JSON,
      checkpoints, predictions, or metric files are staged.
- [ ] Run `python scripts/release_audit.py .` and require `status: PASS`.
- [ ] Tag the exact reviewed release, for example `v1.0-review`.
- [ ] Archive the tag and record its SHA-256 hash in the response letter.
