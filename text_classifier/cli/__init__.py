"""Command-line entry points for the package.

Installed as console scripts (see ``[project.scripts]`` in ``pyproject.toml``):

    text-classifier-train   train a model directory from CSVs
    text-classifier-infer   classify new items with a trained model
    text-classifier-eval    score a trained model against a labeled CSV

Each module exposes a ``main()`` so it works both as a console script and via
``python -m text_classifier.cli.train`` etc. The CLI layer only does
argument parsing, friendly IO, and logging setup; all real work lives in the
application pipelines.
"""
