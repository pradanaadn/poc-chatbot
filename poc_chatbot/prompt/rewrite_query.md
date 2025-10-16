# TASK: Refine a failed search query

## REASON: Initial query retrieved docs on the correct topic but the wrong specific aspect

## GOAL: Create a more specific query to improve retrieval accuracy

## RULES

1. **PRESERVE**: Keep the core subject/entity from `[ORIGINAL_QUERY]`. DO NOT change the topic.
2. **ANALYZE & SPECIFY**: Determine the user's underlying intent (e.g., build, fix, learn, optimize) and add precise keywords.
   - **Intent -> Keywords:**
   - build -> `how-to`, `guide`, `tutorial`, `configuration`, `setup`
   - fix -> `troubleshoot`, `fix`, `error`, `debug`, `resolve`
   - learn -> `concepts`, `architecture`, `explanation`, `best practices`
   - optimize -> `performance`, `security`, `cost`, `scaling`
3. **CONTEXT_RULE**: The `[FAILED_CONTEXT]` shows what to AVOID. Do NOT copy its terms unless they directly match the user's specific intent.
4. **FORMAT**: Output MUST be a concise search phrase, not a full sentence.

### INPUTS

- `[ORIGINAL_QUERY]`: {original_query}
- `[FAILED_CONTEXT]`: {failed_context_summary}

### OUTPUT_FORMAT: A text

```text
"..."
```
