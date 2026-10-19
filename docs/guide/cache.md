# Cache

Pass a directory to cache extractions and screen checks in, and images read before are not paid
for again:

```bash
framelift run samples --profile examples.fantasy_shop:profile \
  --model anthropic:<model-name> --out out --cache .framelift-cache
```

(`Pipeline(..., cache_dir=".framelift-cache")` in Python, `framelift eval --cache` for
evaluations.) An extraction is keyed by the image sent, the model string, the server's URL, the
model settings when there are any and the profile's `fingerprint`, and a screen check by the
shrunk image checked, the model string, the server's URL, the model settings, the screen
description and `Profile.version`. So changing the crop, the model, the server, the model
settings, the instructions, the output model or `Profile.version` reads the image again, and
changing the description or the version checks it again. The URL is there because a model string
does not name the server: `OPENAI_BASE_URL` points `openai-chat:<model>` at another one. Finding
it needs the provider's API key, so a run needs the key even when every image is cached (except
with `OPENAI_BASE_URL` set, which needs none). A framelift release that changes its own prompts
does not reuse older entries. Validators are code and not part of the key: bump `Profile.version`
when you change them, and keep them deterministic.
Only successful results are cached, and only for models given as strings: a `Model` instance
cannot be told apart from another with different settings, so it is not cached, and a warning says so.
Neither are model settings that JSON cannot hold, such as an `httpx.Timeout`.
An answer that needed more retries than the current `max_retries` is not reused.

A cached observation has `"cached": true` and keeps the usage, retries and latency of the run
that paid for it, so evaluation reports still compare models fairly; a screen check taken from the
cache has `"cached": true` in `screen`. The run summary says how many of the extractions came from
the cache. An output that would not come back exactly as it was from its JSON (for
example one with a `Field(exclude=True)` value or a private attribute set) is not cached; fields
excluded from dumps are never written to the cache. Neither is an output with a `set` or
`frozenset` field of two or more items in practice: their order, and so the JSON they dump to,
changes from one process to the next, so the entry a run writes is not found by the next one. Use
a list where the cache matters. Delete the directory to clear it.

On a local server, such as LM Studio or Ollama that `OPENAI_BASE_URL` points the `openai`
provider at, framelift checks that each response names a model starting with the one asked for,
ignoring case. It stops the run with a `UserError` at the first one that does not, before any
retry; an evaluation sends no further cases and raises it at the end, and the CLI prints it and
exits with code 2. Such servers may answer a model they do not have with another one they have
loaded, and its answers would be recorded and cached as the model asked for. Load the model on
the server, or name it as the server does. A server is local when its host is `localhost`, a
private or Tailscale address, a name without dots, or a name under `.local`, `.lan`,
`.home.arpa` or `.ts.net`. Other servers are not checked, as they may answer an alias or a
deployment name with the name of the model behind it (OpenAI, Azure, Google and public gateways
do). A proxy on a local host is checked; for one that answers an alias with another name, such as
LiteLLM reached by a container name, use its own provider (`litellm:<model>`) rather than
`openai-chat:` with `OPENAI_BASE_URL`.
