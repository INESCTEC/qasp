# Publishing QASP releases

The release workflow is `.github/workflows/publish.yml`. It builds a wheel and
source archive from a release tag, checks metadata, then publishes through PyPI
Trusted Publishing. Publishing uses a short-lived identity token rather than a
stored API key. The workflow requires the tagged commit to belong to `main` and
the tag to match the package version.

## First publication

The intended package owner must sign in to PyPI and configure a pending GitHub
publisher at https://pypi.org/manage/account/publishing/ with these exact fields:

- PyPI project name: `qasp`
- GitHub owner: `INESCTEC`
- Repository: `qasp`
- Workflow filename: `publish.yml`
- Environment name: leave blank (this workflow does not use a GitHub environment)

This grants the named repository workflow permission to create and publish the
project. Register it only under the intended owner account. A missing public
project page does not guarantee that PyPI will accept a name; pending publisher
registration or upload is the authoritative check.

See [PyPI's pending-publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release procedure

1. Update the version in `pyproject.toml` and `CITATION.cff`; record the actual
   release date. Retain alpha status until the project adopts a different
   stability policy.
2. Build and check both distributions with `python -m build` and
   `python -m twine check --strict dist/*`.
3. Install the wheel into a clean environment and run
   `python tools/check_compatibility.py --output /tmp/qasp-release-check`.
4. Merge the reviewed release changes into `main`, then create the matching
   version tag and GitHub release. Publishing the release triggers PyPI upload.
5. Confirm that the workflow succeeds and PyPI lists the expected version.
6. In another clean environment, install `qasp==VERSION` explicitly from
   `https://pypi.org/simple`, run `pip check`, verify the import comes from that
   environment, and run the same compatibility checker outside the checkout.

A failed upload can be retried using the workflow's manual dispatch with the
existing tag. PyPI does not allow replacing an uploaded distribution: resolve
an already-published defective release with a new version, rather than changing
the tag or reusing filenames.
