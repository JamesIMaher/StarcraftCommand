FROM registry.access.redhat.com/ubi9/python-311

# UBI's S2I convention: already non-root (uid 1001, gid 0). The platform
# runs the pod as its own uid with a read-only root filesystem anyway, and
# this app writes nothing to disk -- all relay state is in memory.
WORKDIR /opt/app-root/src

COPY --chown=1001:0 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1001:0 app.py ./
COPY --chown=1001:0 static/ ./static/

EXPOSE 8000

# One worker: channels, queues and waiting requests live in this process.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
