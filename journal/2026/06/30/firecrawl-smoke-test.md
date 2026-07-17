# Firecrawl smoke test

## Summary

- `firecrawl_search` worked for a domain-restricted search against `docs.firecrawl.dev`.
- Search query: `Firecrawl search API docs`
- Top result: `https://docs.firecrawl.dev/api-reference/endpoint/search`
- `firecrawl_search_feedback` was attempted, but this deployment returned `DB_DISABLED`.
- `firecrawl_scrape` against the Firecrawl docs page returned only metadata, even with `waitFor: 5000`.
- `firecrawl_scrape` against `https://example.com` worked and returned markdown content.

## Notes

- Search is usable.
- Scraping static pages is usable.
- Scraping Mintlify-rendered Firecrawl docs may need a different URL, map-assisted discovery, or another extraction approach.
