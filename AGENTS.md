# Repository Instructions

## Commit co-authorship

- Commits created by Codex/GPT agents must include a `Co-Authored-By` trailer with the actual model and agent used. The default for GPT-5.6 Terra is:

  ```text
  Co-Authored-By: GPT-5.6 Terra <noreply@openai.com>
  ```

  Adjust the name portion when another model or agent is used, for example `GPT-5.6 Sol`.
- Commits created by Claude must continue to include the corresponding Claude `Co-Authored-By` trailer, for example:

  ```text
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  ```

- Do not rewrite existing history solely to add or change co-author trailers.
