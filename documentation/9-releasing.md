# Publishing a release

This guide is for the maintainer. Current release preparation targets **2.0.0**;
local commits, builds and draft notes do not publish anything by themselves.

## 1. One-time account setup

On PyPI, use the account that owns `atracker`. If it is a new project, register a
pending Trusted Publisher under your account's Publishing settings; otherwise add
one in the existing project's Publishing settings. The name must be available or
owned by you; configuring a pending publisher does not reserve a package name.

| Field | Value |
| --- | --- |
| PyPI project name | `atracker` |
| GitHub owner | `JolleJolles` |
| Repository | `atracker` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Create/verify your PyPI account, verified email and required two-factor
authentication. Do not put passwords or API tokens in the repository. Trusted
Publishing uses GitHub's identity token instead of a stored PyPI token.
See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [first-project setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

In GitHub, create the **pypi** environment under repository Settings → Environments.
Where available, add yourself as a required reviewer so publishing waits for your
approval. Keep the environment name identical to the PyPI publisher configuration.
See [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## 2. Prepare locally

- Finish the intended changes and review the working tree.
- Update `atracker/__version__.py`, `CITATION.cff` version/date and the changelog.
  Keep an empty Unreleased section for subsequent work.
- Review migration guidance and README examples against current signatures.
- Run the tests in the supported working environment, build the source archive
  and wheel, check their metadata and verify the installed wheel imports.

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

Use a fresh `dist` directory for each version so old artifacts are not uploaded.
Test installation of the exact wheel before release; broader operating-system
coverage is a separate roadmap item. Generated `*.egg-info`, `build` and `dist`
are local build outputs and are not committed.

## 3. Publish after reviewing the prepared result

Commit the release preparation, push the intended commits, create the annotated
tag `v2.0.0` at the checked release commit, and push that tag. Those are explicit
publishing steps; local release preparation does not perform them.

Create a GitHub release for the tag. Use the 2.0.0 section of `CHANGELOG.md` for the
release notes, including breaking changes and the migration link. A draft or a
prerelease does not publish to PyPI through the prepared workflow. Publishing a
stable GitHub release starts `.github/workflows/publish.yml`, which checks that
the tag matches the package version, builds/checks artifacts and publishes through
the **pypi** environment. Approve the environment job when GitHub requests it.

Verify the Actions result and PyPI project page, then check:

```bash
python -m pip install --upgrade atracker==2.0.0
python -c 'import atracker; print(atracker.__version__)'
```

If Zenodo's GitHub integration is enabled, check the new archive and its version
DOI after the GitHub release. Retain the existing concept DOI for citations across
versions unless you deliberately want to cite one particular version.

PyPI files cannot be replaced under an existing filename/version. If a published
artifact needs fixing, prepare a new patch version rather than overwriting it.
