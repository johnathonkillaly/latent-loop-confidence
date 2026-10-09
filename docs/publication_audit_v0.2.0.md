# v0.2.0 publication audit

Completed scientific snapshot: `a6ccb384745c623f475e7b4b5fcc9699423b6e31`, descending from Experiment 5 `17ae4e4acac70ef099a55522f0964381a01516d9` and public v0.1.0 `0a637eb97d66eebbd740b612ab92e9095c496931`. Release edits affect README and new/current landing documentation only. Scientific source, reports, configurations and measured artifacts are preserved. The research tag is v0.2.0; historical package metadata remains 0.1.0 to preserve frozen source hashes.

## Scientific checks

All requested reference values match recorded data: 18,884 parameters; grid four/eight-loop accuracy 98.28/99.56%; four-minus-one +36.44 pp [32.55,40.50]; feedforward 97.72/98.33%; eight-loop 7×7/long-path completion 95/54.44%; Qwen recurrent/one-pass 28.33/47.78%; Snake four-loop horizon-six 96.61%, matched by four-block feedforward. Three-seed paired intervals cluster shared instances. Within-model depth gains, architecture superiority and transferable planning are distinct claims. Only the first has strong supporting evidence here.

Fresh CPU-only regeneration of the E5 and E5B reports succeeded. E5 audited 1,584 labels, 25,320 trajectory examples and 86,784 state metrics. E5B audited 6,840 maze labels/layouts, 290,640 grid/Qwen decisions, 4,800 Snake states and 18,000 neural Snake decisions. Independent solver verification and failed variants remain public. Original historical report families regenerated in separate checkouts. Charts read saved measurements; no invented values. No independent complete training replication is claimed.

Release test rerun: **48 passed with MLX integration**; **31 passed, 13 skipped in a clean CPU-only environment**. Dependency compatibility check passes for 50 installed packages. CLI entry points work. Existing completed-run verification passed checkpoint/source hashes and 72 trained locality checks before release documentation edits. Its historical-path assertion is designed for the original scientific snapshot; run it at `a6ccb38`, rather than interpreting new release documentation as scientific mutation. Report-only audits remain available at this release. One attempted nonexistent `public-audit` CLI command was rejected; the supported `report` command successfully performs that audit.

## Security, privacy and licensing

The all-local-reference history scan inspected **1,216 blobs**, decompressing **614 gzip blobs**, across 22 local commits. No file exceeded 10 MiB; no model-weight/tensor/credential artifact extension was found. Three content flags were manually resolved: two Python matrix-multiplication expressions mistaken for email addresses and one scanner's own private-path pattern. No actual credential/private-path finding remained. This is a heuristic review aided by manual source, log, metadata, generated-asset and provenance inspection, not a guarantee against every possible secret.

Only the canonical completed ancestry is published (19 commits before release documentation). Its author/committer identities use the project's generic research identity. Other local refs, including preserved prepublication identity metadata, are not pushed. Original verified Git bundles remain outside the repository; no history rewrite occurred. A previously reviewed Qwen protocol path-key privacy amendment is retained with original content hashes and an ignored original receipt. Neither original private metadata nor ignored environments, caches, checkpoints or hidden tensors is included.

Synthetic environments/data and grid/Snake implementations are original project code under MIT. No third-party implementation is vendored. Qwen3's pinned official license was rechecked as Apache-2.0; separately downloaded weights/tokenizer are excluded. MLX's official MIT license and dependency provenance are documented in [third-party attribution](third_party_licenses.md). Papers are linked/paraphrased, not redistributed. RecurTrace inspiration is not a reproduction claim.

GitHub destination was verified as the existing public `johnathonkillaly/latent-loop-confidence`, default `main`, at the original v0.1.0 commit. Publication uses a normal fast-forward and a new annotated tag, preserving unrelated historical research branches and v0.1.0. Experiment 5.2 is excluded. No critical publication blocker was found.

[Release notes](release_notes_v0.2.0.md) · [Reproduction scope](reproduction_v0.2.0.md)
