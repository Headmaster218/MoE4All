# Compact Benchmark Index

Trusted main CI rebuilds `v1/` after benchmark originals change. Report PRs must
contain only one original `.m4b`; contributors must not submit generated files.

`v1/manifest.json` is the fixed browser entry. Its `package_commit` pins both
content-addressed gzip/Brotli packages to an immutable Git tree. CI commits the
packages first, then the manifest, and pushes both commits together without force.
The latest tree keeps one package pair; historical manifests remain readable from
their pinned commits. Repeating the same input creates no new commit.

Data source commits and package publication commits are distinct. Originals,
compressed payloads, columns, source links, sizes and SHA-256 are validated before
publication. No Cloudflare request or website deployment is involved.

Use the workflow's manual entry for initial publication or retry. Repository rules
must permit its bot to update these generated files; protection is never bypassed.
