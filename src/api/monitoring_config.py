"""Generate Prometheus configuration using the runtime API key, never a committed key."""
import json
import os
from pathlib import Path


def main():
    config = {
        "global": {"scrape_interval": "15s"},
        "scrape_configs": [{
            "job_name": "adult-governance", "metrics_path": "/metrics/prometheus",
            "http_headers": {"X-API-Key": {"secrets": [os.environ["API_KEY"]]}},
            "static_configs": [{"targets": ["api:5000"]}],
        }],
    }
    Path("/monitoring/prometheus.yml").write_text(json.dumps(config))


if __name__ == "__main__":
    main()
