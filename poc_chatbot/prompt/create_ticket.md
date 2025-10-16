# TASK: Generate a structured support ticket from a conversation transcript

## OUTPUT_FORMAT: A single, complete JSON object. NO other text, explanation, or conversation

## JSON SCHEMA

{
"title": "string", // A concise summary. Max 200 characters.
"description": "string", // A detailed, structured summary of the issue.
"labels": ["string"], // An array of relevant keywords/tags.
"priority": "string" // Must be one of: "low", "medium", "high", "urgent".
}

---

### FIELD GENERATION LOGIC

**1. `title`:**
_Summarize the core problem in a single, clear sentence.
_ If a specific product or feature is mentioned, include it. \* STRICTLY adhere to the 200-character limit.

**2. `description`:**
_Structure the description using these subheadings:
_ **User's Reported Problem:** What did the user report in their own words (summarized)?
_**Steps Taken:** What has the user already tried to do to solve the issue (if mentioned)?
_ **Expected vs. Actual Behavior:** What did the user expect to happen versus what actually happened? \* If a critical piece of information is missing and you must make an assumption to make the ticket coherent, you MUST state it clearly, like this: `[ASSUMPTION: Assuming the user is on the 'Production' environment.]`

**3. `labels`:**
_Extract key technical terms, product names, and feature names from the conversation.
_ Include the user's intent if clear (e.g., `bug_report`, `feature_request`, `how-to_question`). \* Generate a minimum of 2 and a maximum of 5 labels.

**4. `priority` (Use this strict decision tree):**
_**`urgent`**: If the issue is a security vulnerability, a full system outage, or is causing a critical business impact for the user.
_ **`high`**: If a core feature is broken for many users, there is no workaround, or significant data loss is occurring.
_**`medium`**: If a non-critical feature is malfunctioning, there is a viable workaround, or the user is experiencing a significant inconvenience.
_ **`low`**: If the issue is a cosmetic bug, a general "how-to" question, a feature request with no immediate impact, or has minimal user impact.

---

### CRITICAL SAFETY RULE

- **PII REDACTION:** You MUST redact all Personally Identifiable Information (PII). This includes, but is not limited to: names, email addresses, phone numbers, physical addresses, IP addresses, specific user IDs, and API keys/tokens. Replace them with a placeholder like `[REDACTED]`.

---

### INPUT

[CONVERSATION_TRANSCRIPT]
{conversation_text}
