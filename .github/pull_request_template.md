<!--

Thank you for pull request.

Below are a few things we ask you kindly to self-check before getting a review. Remove checks that are not relevant.

-->


Checklist
* [ ] The documentation examples have been regenerated if the Jupyter notebooks in the `examples/` have changed. To regenerate the documentation examples, run `jupyter nbconvert --to rst --output-dir=docs/examples examples/*.ipynb` from the top level directory)
* [ ] If any dependencies have changed, the changes are reflected in the
  * [ ] `setup.py` (runtime requirements)
  * [ ] `pyproject.toml` (build requirements)
  * [ ] `requirements-dev.txt` (build and development requirements)
