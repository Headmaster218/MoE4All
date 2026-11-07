# Compact Benchmark Index

Trusted main CI rebuilds `v1/` on main after benchmark originals change. Report PRs must
contain only one original `.m4b`; contributors must not submit generated files.

`v1/manifest.json` is the fixed browser entry. Its `package_commit` pins both
content-addressed gzip/Brotli packages to an immutable Git tree. CI commits the
packages first, then the manifest, retaining both commits in an automatic index-only merge.
The latest tree keeps one package pair; historical manifests remain readable from
their pinned commits. Repeating the same input creates no new commit.

Data source commits and package publication commits are distinct. Originals,
compressed payloads, columns, source links, sizes and SHA-256 are validated before
publication. No Cloudflare request or website deployment is involved.

Use the workflow's manual entry for initial publication or retry. Repository Actions
must be allowed to create PRs. The trusted publisher automatically merges only its
own freshly generated index PR after verifying its actor, repository, exact head,
allowed files and main freshness. Source report PRs are never auto-approved or
auto-merged. Main keeps its protection; no permanent data branch or second human
approval is required. Generated-only commits skip redundant general CI after the
publisher's own validation. Normal report PR validation is unchanged.
