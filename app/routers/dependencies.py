"""Shared validation for local HTTP controls; no route registration."""
from typing import Annotated, Literal
from urllib.parse import urlparse

from fastapi import HTTPException, Path
from pydantic import BaseModel, ConfigDict

Id = Annotated[str, Path(pattern=r"^[0-9a-f]{24}$")]

class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Literal["pause", "stop", "resume"]


def mutation_origin(request):
    origin = request.headers.get("origin")
    if origin:
        host = urlparse(origin).hostname
        local = {"127.0.0.1", "localhost", "::1"}
        if host != request.url.hostname and not (host in local and request.url.hostname in local):
            raise HTTPException(403, "Cross-origin annotation control rejected")

