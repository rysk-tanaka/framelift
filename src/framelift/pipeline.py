"""`Pipeline`: run a `Profile` over images and write ``observations.jsonl``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_ai.exceptions import UnexpectedModelBehavior

from framelift._internal.extract import Extractor, model_label
from framelift._internal.sources import expand_inputs, load_image
from framelift.observation import Observation, ProfileRef, SourceRef, Usage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic_ai.models import Model

    from framelift._internal.sources import ImageSource
    from framelift.profile import Profile

OBSERVATIONS_FILE = "observations.jsonl"


@dataclass
class RunResult:
    observations: list[Observation] = field(default_factory=list)
    output_path: Path | None = None

    @property
    def succeeded(self) -> list[Observation]:
        return [o for o in self.observations if o.status == "ok"]

    @property
    def failed(self) -> list[Observation]:
        return [o for o in self.observations if o.status == "error"]


class Pipeline[OutputT: BaseModel]:
    """Extract ``profile.output_type`` from every input image.

    Args:
        profile: What to read.
        model: A Pydantic AI model or a ``"provider:model"`` string.
        out_dir: Where to write ``observations.jsonl``. ``None`` keeps results in memory only.
        max_retries: How many times a failed validation may ask the model to try again.
        anonymize_paths: Omit input paths from observations, keeping only hashes.
    """

    def __init__(
        self,
        profile: Profile[OutputT],
        *,
        model: Model | str,
        out_dir: str | Path | None = None,
        max_retries: int = 2,
        anonymize_paths: bool = False,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.profile = profile
        self.model = model
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.anonymize_paths = anonymize_paths
        self._extractor = Extractor(profile, model, max_retries=max_retries)

    def run(self, inputs: str | Path | Iterable[str | Path]) -> RunResult:
        """Process the inputs in order. A failure on one image does not stop the run."""
        items = [inputs] if isinstance(inputs, str | Path) else list(inputs)
        paths = expand_inputs(items)

        result = RunResult()
        writer = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            result.output_path = self.out_dir / OBSERVATIONS_FILE
            writer = result.output_path.open("w", encoding="utf-8")
        try:
            for path in paths:
                observation = self._observe(load_image(path))
                result.observations.append(observation)
                if writer is not None:
                    writer.write(observation.to_json() + "\n")
                    writer.flush()
        finally:
            if writer is not None:
                writer.close()
        return result

    def _observe(self, image: ImageSource) -> Observation:
        profile = ProfileRef(name=self.profile.name, version=self.profile.version)
        source = SourceRef(
            path=None if self.anonymize_paths else image.path.as_posix(),
            sha256=image.sha256,
            phash=image.phash,
        )
        model = model_label(self.model)
        started = time.perf_counter()
        try:
            extraction = self._extractor.extract(image)
        except UnexpectedModelBehavior as exc:
            return Observation(
                status="error",
                profile=profile,
                source=source,
                data=None,
                error=str(exc),
                model=model,
                usage=Usage(latency_ms=round((time.perf_counter() - started) * 1000)),
                retries=0,
            )
        return Observation(
            status="ok",
            profile=profile,
            source=source,
            data=extraction.output.model_dump(mode="json"),
            model=model,
            usage=Usage(
                input_tokens=extraction.input_tokens,
                output_tokens=extraction.output_tokens,
                requests=extraction.requests,
                latency_ms=extraction.latency_ms,
            ),
            retries=extraction.retries,
        )
