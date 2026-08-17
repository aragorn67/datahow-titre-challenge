# The inference service, as a self-contained image.
#
# Deliberately simple. A multi-stage build is the usual reflex, and it earns its
# complexity when a compiler is needed at build time and unwanted at run time.
# Nothing here compiles: numpy and scipy install from prebuilt wheels, so a second
# stage would remove pip's cache and little else -- which `--no-cache-dir` already
# does. Simple and explicable beats clever and unexamined.
#
# Build:  docker build -t titre-predictor .
# Run:    docker run --rm -p 8000:8000 titre-predictor
# Check:  curl http://localhost:8000/health

FROM python:3.13-slim

# Python behaviour inside a container:
#   DONTWRITEBYTECODE -- .pyc files are wasted layers; the image is never reused
#                        for a second run of the same interpreter
#   UNBUFFERED        -- logs reach `docker logs` immediately rather than sitting
#                        in a buffer, which matters when diagnosing a crash
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TITRE_MODEL_PATH=/app/artefacts/titre_model.json

WORKDIR /app

# Dependencies are installed before the source is copied. Docker caches each step
# and invalidates everything after the first change, so this ordering means editing
# a source file rebuilds in seconds rather than re-downloading numpy and scipy.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[service]"

# The fitted model is baked in rather than mounted at run time. The image is then a
# complete, versioned unit: `docker run` serves predictions with no further setup,
# and the model that was validated is the model that ships. The alternative --
# mounting a volume -- allows swapping models without a rebuild, at the cost of an
# image that cannot serve on its own and a missing mount becoming a runtime failure.
# TITRE_MODEL_PATH above stays configurable, so mounting elsewhere remains possible
# for anyone who wants it.
COPY artefacts/titre_model.json ./artefacts/titre_model.json

# Run as a non-root user. A container process that is root is root on the host
# kernel if it escapes, and nothing here needs privileges: the service reads one
# file and listens on a port above 1024.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The payoff for making /health a readiness check rather than a liveness one: this
# reports unhealthy until the model has actually loaded, so an orchestrator holds
# traffic back instead of sending it to a container that cannot answer.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. Under the shell
# form a shell would be PID 1, absorb the signal, and `docker stop` would wait the
# full timeout before killing the container instead of shutting down cleanly.
CMD ["uvicorn", "titre_predictor.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
