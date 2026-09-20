# Anonymous source export

The release was assembled from an explicit source-file allowlist. Original Git
history, remotes, manuscript files, local notes, credentials, cluster scripts,
generated responses, experiment outputs, and model/tokenizer binaries were
excluded.

Personal credential aliases, request application labels, and hardcoded cloud
configuration were replaced with generic names or environment settings.
Baseline API key files are optional where environment credentials suffice.
The GPTFuzz adapters no longer inject a cluster-specific Python installation.
Standalone script imports were corrected where needed for the documented commands.
Scientific attribution and the bundled upstream license are retained.

The controller, gate equations, distribution reconstruction, and optimization
objectives are preserved. The portable count-recovery helper exposes the
existing job-script conversion. Tests that previously required local benchmark
files now use temporary synthetic records. One test for excluded Slurm scripts
is explicitly skipped.

`docs/source_export.json` lists adjusted source files. `MANIFEST.sha256` covers
the distributable files. Run `python scripts/audit_release.py` to check syntax,
credential patterns, local paths, and manifest integrity. To add identifiers
specific to your environment, use repeatable `--deny` arguments; the scanner
prints finding locations, not matched values. After intentionally editing a
release, use `--write-manifest` to refresh its checksums.

The generated source ZIP contains no `.git` directory, caches, or file-owner
metadata. A local Git checkout, when provided, starts with a fresh anonymous
commit and has no configured remote.
