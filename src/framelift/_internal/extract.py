"""Structured extraction from one image with Pydantic AI."""

from __future__ import annotations

import ipaddress
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel
from pydantic_ai import Agent, AgentRetries, BinaryContent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.models import infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.usage import RunUsage

from framelift._internal.cache import ExtractEntry, cache_key, json_sha256, same_state
from framelift.observation import FrameRef, Observation, ProfileRef, SourceRef, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, StreamedResponse
    from pydantic_ai.settings import ModelSettings

    from framelift._internal.cache import Cache
    from framelift._internal.sources import ImageSource
    from framelift.observation import Crop, Screen
    from framelift.profile import Profile, Validator

USER_PROMPT = "Extract the data shown in this screenshot."


@dataclass(frozen=True)
class Extraction[OutputT: BaseModel]:
    output: OutputT | None
    """The validated output, or None when validation failed after every retry."""
    error: str | None
    input_tokens: int
    output_tokens: int
    requests: int
    retries: int
    latency_ms: int
    cached: bool = False
    """Read from the cache: the usage, retries and latency are those of the original run."""


def _adapt[OutputT: BaseModel](validator: Validator[OutputT]):
    def run(output: OutputT) -> OutputT:
        result = validator(output)
        return output if result is None else result

    return run


def model_label(model: Model | str) -> str:
    if isinstance(model, str):
        return model
    return f"{model.system}:{model.model_name}"


class ModelMismatchError(UserError):
    """A local server answered with another model than the one asked for."""


# Tailscale's addresses (shared address space), which `is_private` leaves out.
_SHARED_ADDRESSES = ipaddress.ip_network("100.64.0.0/10")
# Names that only a home network or a tailnet resolves. Not ".internal": clouds and companies
# name their own services, gateways among them, under it.
_LOCAL_SUFFIXES = (".localhost", ".local", ".lan", ".home.arpa", ".ts.net")


def _is_local_host(host: str | None) -> bool:
    if host is None:
        return False
    host = host.rstrip(".")  # a fully qualified name, such as "lmstudio.local."
    is_local_name = host == "localhost" or host.endswith(_LOCAL_SUFFIXES)
    # A bare name such as "m6m" comes from the hosts file or the local network's DNS.
    is_bare_name = "." not in host and ":" not in host
    if is_local_name or is_bare_name:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            # IPv4 shorthand such as "127.1", which the OS connects to as 127.0.0.1.
            address = ipaddress.ip_address(socket.inet_aton(host))
        except OSError:  # a public name, such as a provider's or a cloud server's
            return False
    return address.is_private or address in _SHARED_ADDRESSES


def is_local_server(model: Model) -> bool:
    """Whether ``model`` asks a server on this machine or the local network through the OpenAI
    API, as ``OPENAI_BASE_URL`` sets one for LM Studio, say.

    Only such servers are checked: they are the ones that answer a model they do not have with
    one they have loaded. Providers and proxies elsewhere may answer an alias or a deployment
    name with the name of the model behind it, so their names need not match.
    """
    base_url = model.base_url
    if model.system != "openai" or base_url is None:
        return False
    return _is_local_host(urlsplit(base_url).hostname)


def _bare_name(name: str) -> str:
    """``name`` without what a server may add or drop for the same model: the publisher
    ("openai/gpt-oss-20b"), the directories of a model file's path (Windows ones too), the
    quantization asked for ("qwen3-vl-8b@q4_k_m") and the case."""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return name.split("@", 1)[0].casefold()


def check_model_name(model: Model, response: ModelResponse) -> None:
    """Raise `ModelMismatchError` when ``response`` names a model other than ``model``'s.

    The name answered only has to start with the name asked, ignoring case, as a server may
    answer an alias with a dated name. Both are compared by `_bare_name`.
    """
    answered = response.model_name
    # Some servers leave the name empty, which says no more than leaving it out.
    if not answered:
        return
    asked = _bare_name(model.model_name)
    if asked:
        is_model_asked = _bare_name(answered).startswith(asked)
    else:
        # A name that is all publisher or quantization is compared whole, both sides: every name
        # starts with an empty one.
        is_model_asked = answered.casefold().startswith(model.model_name.casefold())
    if is_model_asked:
        return
    # No URL in the message: it may carry credentials.
    raise ModelMismatchError(
        f"Asked for model {model.model_name!r}, but the server answered with {answered!r}; it "
        f"may have used another model it had loaded. Load {model.model_name!r} on the server, "
        "or name the model as the server does. A local proxy that answers an alias with another "
        "name is not checked when asked through its own provider, such as 'litellm:<model>'"
    )


class _NameChecked(WrapperModel):
    """``wrapped``, checking the model named in each response, so that a server answering with
    another model is stopped at its first answer instead of after every retry is paid for.

    Streamed requests are checked once read: Pydantic AI streams a run that has capabilities
    or an event stream handler, even when framelift does not ask for a stream.
    """

    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        response = await super().request(*args, **kwargs)
        check_model_name(self.wrapped, response)
        return response

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[StreamedResponse]:
        async with super().request_stream(*args, **kwargs) as stream:
            yield stream
        # Once read: a streamed response may name its model only partway through.
        check_model_name(self.wrapped, stream.get())


def resolve_model(model: Model | str) -> Model:
    """``model`` as a `Model`, checking the names answered when it is a local server (see
    `is_local_server`), whose answers from another model would otherwise be recorded and cached
    as the model asked for."""
    resolved = infer_model(model)
    return _NameChecked(resolved) if is_local_server(resolved) else resolved


def note_input(exc: BaseException, image: ImageSource, *, anonymize_paths: bool) -> None:
    """Add to ``exc`` which input it was found in: its path, or its hash when paths are
    anonymized, as in the observations."""
    where = image.source_sha256 if anonymize_paths else image.path.as_posix()
    # A video's frame too, so that the observations written before it can be told apart.
    is_video_frame = image.timestamp_ms is not None
    if is_video_frame:
        where += f" at frame {image.frame_index} ({image.timestamp_ms} ms)"
    exc.add_note(f"Found while reading {where}")


def _prompt(image: ImageSource) -> list[str | BinaryContent]:
    return [USER_PROMPT, BinaryContent(data=image.data, media_type=image.media_type)]


def _extraction[OutputT: BaseModel](
    output: OutputT | None,
    error: str | None,
    usage: RunUsage,
    messages: list[ModelMessage],
    started: float,
) -> Extraction[OutputT]:
    retries = sum(
        isinstance(part, RetryPromptPart) for message in messages for part in message.parts
    )
    return Extraction(
        output=output,
        error=error,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
        retries=retries,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


class Extractor[OutputT: BaseModel]:
    """Reads one image into ``profile.output_type``, retrying when validation fails."""

    def __init__(
        self,
        profile: Profile[OutputT],
        model: Model | str,
        *,
        max_retries: int,
        cache: Cache | None = None,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self._profile = profile
        self._label = model_label(model)
        # Resolved once, so the server in the cache key is the one asked.
        self._model = resolve_model(model)
        self._max_retries = max_retries
        self._cache = cache
        self._model_id = (
            None if cache is None else cache.model_id(model, self._model, model_settings)
        )
        # Computed once: it builds the output model's JSON schema.
        self._fingerprint = profile.fingerprint if self._model_id is not None else ""
        self._agent: Agent[None, OutputT] = Agent(
            output_type=profile.output_type,
            instructions=profile.instructions,
            retries=AgentRetries(output=max_retries),
            name=f"framelift:{profile.name}",
            model_settings=model_settings,
        )
        for validator in profile.validators:
            self._agent.output_validator(_adapt(validator))

    def extract(self, image: ImageSource) -> Extraction[OutputT]:
        """Read one image, or take it from the cache. A failure after every retry is returned,
        not raised, with the usage and retries it took, so failed images still count toward
        cost. A local server answering with another model raises `ModelMismatchError`, as every
        image would be read by it."""
        key = self._cache_key(image)
        cached = self._cached(key)
        if cached is not None:
            return cached
        extraction = self._extract(image)
        self._store(key, extraction)
        return extraction

    async def extract_async(self, image: ImageSource) -> Extraction[OutputT]:
        """`extract` for callers that already run an event loop, such as pydantic-evals."""
        key = self._cache_key(image)
        cached = self._cached(key)
        if cached is not None:
            return cached
        extraction = await self._extract_async(image)
        self._store(key, extraction)
        return extraction

    def _extract(self, image: ImageSource) -> Extraction[OutputT]:
        started = time.perf_counter()
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = self._agent.run_sync(_prompt(image), model=self._model, usage=usage)
            except UnexpectedModelBehavior as exc:
                return _extraction(None, str(exc), usage, messages, started)
        return _extraction(result.output, None, usage, messages, started)

    async def _extract_async(self, image: ImageSource) -> Extraction[OutputT]:
        started = time.perf_counter()
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = await self._agent.run(_prompt(image), model=self._model, usage=usage)
            except UnexpectedModelBehavior as exc:
                return _extraction(None, str(exc), usage, messages, started)
        return _extraction(result.output, None, usage, messages, started)

    def _cache_key(self, image: ImageSource) -> str | None:
        if self._model_id is None:
            return None
        return cache_key(image.sha256, *self._model_id, self._fingerprint)

    def _cached(self, key: str | None) -> Extraction[OutputT] | None:
        if key is None or self._cache is None:
            return None
        entry = self._cache.read("extract", key, ExtractEntry)
        # One that took more retries than allowed now would not have succeeded under this limit.
        if entry is None or entry.retries > self._max_retries:
            return None
        # Anything the output model's own validators or serializers raise is a miss: the cache
        # must never stop a run that would succeed without it.
        try:
            output = self._profile.output_type.model_validate_json(entry.input, by_name=True)
            if json_sha256(output.model_dump(mode="json")) != entry.data_sha256:
                return None
        except Exception:
            return None
        return Extraction(
            output=output,
            error=None,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            requests=entry.requests,
            retries=entry.retries,
            latency_ms=entry.latency_ms,
            cached=True,
        )

    def _store(self, key: str | None, extraction: Extraction[OutputT]) -> None:
        # Failures are not stored: the next run may succeed.
        output = extraction.output
        if key is None or self._cache is None or output is None:
            return
        # By field name, so a model that dumps by alias still validates back, and only the
        # fields that were set, so they are the ones set again. Fields excluded from dumps and
        # private attributes are not stored, so what observations leave out stays out of the
        # cache too.
        try:
            text = output.model_dump_json(by_alias=False, round_trip=True, exclude_unset=True)
            restored = self._profile.output_type.model_validate_json(text, by_name=True)
        except Exception:  # e.g. a validator that reads an alias key; not stored, as above
            return
        # Only an output that comes back exactly as it was is stored, so a hit gives the same
        # data, items and keys as the run that paid for it. One that does not (an excluded field
        # or private attribute set, a validator that changes its input, NaN) is just not cached.
        try:
            is_same = same_state(restored, output)
        except Exception:  # e.g. RecursionError on a private attribute that refers to itself
            return
        if not is_same:
            return
        entry = ExtractEntry(
            input=text,
            data_sha256=json_sha256(output.model_dump(mode="json")),
            input_tokens=extraction.input_tokens,
            output_tokens=extraction.output_tokens,
            requests=extraction.requests,
            retries=extraction.retries,
            latency_ms=extraction.latency_ms,
        )
        self._cache.write("extract", key, entry)

    def observe(
        self, image: ImageSource, *, anonymize_paths: bool = False, screen: Screen | None = None
    ) -> tuple[Observation, OutputT | None]:
        """Extract one image into an `Observation`; a failed extraction becomes an error record,
        and another model answering raises `ModelMismatchError` (see `extract`).

        Also returns the validated output (``None`` on error), so callers that need the model,
        such as evaluation, do not have to validate the stored JSON again. ``screen`` is the
        screen check that let the image through, if one was asked.
        """
        extraction = self.extract(image)
        return self._observation(image, extraction, anonymize_paths, screen), extraction.output

    async def observe_async(
        self, image: ImageSource, *, anonymize_paths: bool = False
    ) -> tuple[Observation, OutputT | None]:
        """`observe` for callers that already run an event loop."""
        extraction = await self.extract_async(image)
        return self._observation(image, extraction, anonymize_paths), extraction.output

    def unreadable(
        self,
        path: Path,
        sha256: str | None,
        error: str,
        *,
        anonymize_paths: bool = False,
        crop: Crop | None = None,
    ) -> Observation:
        """An error record for an input that could not be read, or a screen that could not be
        cropped (``crop`` then says its orientation), so nothing was sent."""
        return Observation(
            status="error",
            profile=ProfileRef(name=self._profile.name, version=self._profile.version),
            source=SourceRef(path=None if anonymize_paths else path.as_posix(), sha256=sha256),
            frame=None,
            crop=crop,
            screen=None,
            cached=False,
            data=None,
            error=error,
            model=self._label,
            usage=Usage(),
            retries=0,
        )

    def screened_out(
        self, image: ImageSource, screen: Screen, *, anonymize_paths: bool = False
    ) -> Observation:
        """The record of an image not extracted because of its screen check: skipped when the
        check found another screen, an error when the check itself failed."""
        return self._observation(
            image,
            Extraction(
                output=None,
                error=None if screen.error is None else f"Screen check failed: {screen.error}",
                input_tokens=0,
                output_tokens=0,
                requests=0,
                retries=0,
                latency_ms=0,
            ),
            anonymize_paths,
            screen,
            status="error" if screen.error is not None else "skipped",
        )

    def _observation(
        self,
        image: ImageSource,
        extraction: Extraction[OutputT],
        anonymize_paths: bool,
        screen: Screen | None = None,
        *,
        status: Literal["skipped", "error"] | None = None,
    ) -> Observation:
        output = extraction.output
        return Observation(
            status=status or ("ok" if output is not None else "error"),
            profile=ProfileRef(name=self._profile.name, version=self._profile.version),
            source=SourceRef(
                path=None if anonymize_paths else image.path.as_posix(),
                sha256=image.source_sha256,
            ),
            frame=FrameRef(
                index=image.frame_index,
                timestamp_ms=image.timestamp_ms,
                sha256=image.sha256,
                phash=image.phash,
            ),
            crop=image.crop,
            screen=screen,
            cached=extraction.cached,
            data=None if output is None else output.model_dump(mode="json"),
            error=extraction.error,
            model=self._label,
            usage=Usage(
                input_tokens=extraction.input_tokens,
                output_tokens=extraction.output_tokens,
                requests=extraction.requests,
                latency_ms=extraction.latency_ms,
            ),
            retries=extraction.retries,
        )
