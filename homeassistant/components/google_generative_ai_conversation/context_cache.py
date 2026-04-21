"""Explicit context caching for the Google Generative AI integration.

This module is a scaffold for the explicit context caching feature discussed in
the tracking issue. Nothing in here is wired up yet: the manager is created but
``async_get_cache_name`` always returns ``None``, so the integration keeps
relying on implicit caching on the Gemini side.

Planned responsibilities once the design is agreed with upstream maintainers:
- Create a ``CachedContent`` per subentry covering the system instruction plus
  the tool declarations, but only when the token count clears the per-model
  minimum (1024 on 2.5 Flash, 2048 on 2.5 Pro).
- Invalidate and rebuild the cache when the prompt, the tool set, or the model
  changes, using a content hash.
- Refresh the TTL before expiry to avoid cold calls.
- Expose an opt-out so users on free-tier API keys are never billed for cache
  storage.

See ``_async_handle_chat_log`` in ``entity.py`` for the integration point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google.genai import Client
from google.genai.types import Tool

from .const import LOGGER


@dataclass(slots=True)
class CacheEntry:
    """Bookkeeping for a single cached content object."""

    name: str
    """Server-side cache resource name, passed as ``cached_content``."""

    content_hash: str
    """Hash of the cached payload, used to decide when to rebuild."""


@dataclass(slots=True)
class ContextCacheManager:
    """Manage explicit ``CachedContent`` objects per subentry.

    The manager is attached to the config entry's runtime data. It is keyed by
    subentry id because each subentry (conversation, ai_task, stt) can target
    a different model and tool set.
    """

    client: Client
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    async def async_get_cache_name(
        self,
        subentry_id: str,
        model_name: str,
        system_instruction: str,
        tools: list[Tool] | None,
    ) -> str | None:
        """Return the cache resource name to pass as ``cached_content``.

        Returns ``None`` until the feature is implemented. When implemented,
        returns ``None`` also when the payload is under the model minimum or
        when the user has opted out.
        """
        # TODO: hash(system_instruction + tools + model_name), look up / create
        #   a CachedContent via ``self.client.aio.caches.create``, handle TTL
        #   refresh, store the result in ``self.entries``.
        LOGGER.debug(
            "Context cache lookup for subentry %s skipped (feature not enabled)",
            subentry_id,
        )
        return None

    async def async_invalidate(self, subentry_id: str) -> None:
        """Drop the cache entry for a subentry, e.g. on options update."""
        # TODO: call ``self.client.aio.caches.delete(name=...)`` and pop
        #   ``self.entries[subentry_id]``.
        self.entries.pop(subentry_id, None)
