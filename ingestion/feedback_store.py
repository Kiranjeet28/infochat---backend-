"""Answer feedback storage (thumbs up/down). Never receives the question or the answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SupabaseFeedbackStore:
    project_url: str
    service_role_key: str
    timeout_seconds: int = 3

    def add(self, row: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.project_url.rstrip('/')}/rest/v1/feedback",
            headers={
                "apikey": self.service_role_key,
                "Authorization": f"Bearer {self.service_role_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=row,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
