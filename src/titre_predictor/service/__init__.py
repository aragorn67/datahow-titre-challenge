"""The inference microservice: ``GET /health`` and ``POST /predict``.

Kept inside the ``titre_predictor`` package rather than beside it, because the
service is not a separate program that happens to use the model -- it *is* the
model, exposed over HTTP. Sharing the package is what lets ``app.py`` import
``features`` directly, which is the mechanism that makes a served number provably
identical to a fitted one.

Nothing in the rest of the package imports this subpackage, so the base install
never needs FastAPI. ``tests/test_dependencies.py`` enforces that direction.
"""
