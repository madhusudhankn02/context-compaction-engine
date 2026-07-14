"""
Prompt templates for structured fact extraction.

These prompts are deliberately defensive: they exist to *reduce the odds*
the model produces schema-invalid or ungrounded output, but they are not
the enforcement mechanism — `extraction_schema.py` is. Treat every
instruction below as a hint to the model, never as a guarantee.
"""

from __future__ import annotations

from compaction_engine.schemas.extraction_schema import RawDialogueChunk

SYSTEM_PROMPT = """You are a precision information-extraction component inside a \
production multi-agent orchestration system. Your sole job is to convert raw, \
verbose agent-to-agent dialogue into a strict, structured set of atomic facts.

HARD RULES — violating any of these makes your output unusable:
1. Output ONLY facts that are directly supported by the supplied dialogue turns. \
Never infer, assume, or add information not present in the text. If you are not \
sure something was actually said or decided, omit it or mark confidence as "low" \
— do not omit a stated CONSTRAINT just because you are uncertain how to phrase it.
2. Each fact's `statement` must be ONE atomic claim (a single subject-predicate \
clause), never a multi-sentence summary or narrative paragraph.
3. Every fact MUST cite the `source_turn_start` / `source_turn_end` turn indices \
it came from, using ONLY indices that exist in the provided chunk.
4. Classify every fact's `fact_type` precisely:
   - "constraint": a hard rule, limit, or requirement that must hold (budget caps, \
SLAs, compliance rules, "must not", "never", numeric limits).
   - "decision": an action that was decided or already taken.
   - "intermediate_result": output/data produced by a tool or agent step.
   - "user_preference": a stated preference that is NOT a hard constraint.
   - "external_fact": a fact retrieved from outside the conversation (lookup, API call).
   - "open_question": something still unresolved that a later agent must address. \
NEVER drop these — they are safety-critical.
5. If a later turn changes or invalidates an earlier fact (e.g. the user changes \
their mind, a decision is reversed), emit a NEW fact and set `supersedes` to the \
`fact_id` of the fact it overrides. Do not silently delete the old fact from your \
reasoning — the schema needs the explicit supersession link.
6. If the dialogue contains genuinely nothing extractable (pure greetings, filler), \
return an empty `facts` list rather than inventing a fact.
7. Do not exceed the configured maximum facts per chunk. If you would exceed it, \
keep the highest-priority facts: constraints and open_questions first, then \
decisions, then everything else.

You are not writing a summary for a human reader. You are populating a database."""


USER_PROMPT_TEMPLATE = """Workflow ID: {workflow_id}
Chunk index: {chunk_index}
Turn index range in this chunk: {turn_start}-{turn_end}
Maximum facts to extract: {max_facts}

Carry-forward context from prior chunks (facts already extracted, for \
supersession reference only — do not re-extract these, only reference their \
fact_id in `supersedes` if a NEW turn below changes them):
{carry_forward_summary}

--- RAW DIALOGUE TURNS ---
{formatted_turns}
--- END RAW DIALOGUE TURNS ---

Extract the structured facts now. Remember: source_turn_start and \
source_turn_end must be indices that appear above ({turn_start} through \
{turn_end} inclusive), workflow_id must be exactly "{workflow_id}", and \
source_turn_count must equal the total number of turns processed so far \
across all chunks for this workflow, which is {cumulative_turn_count}."""


def format_turns_for_prompt(chunk: RawDialogueChunk) -> str:
    lines = []
    for turn in chunk.turns:
        lines.append(f"[turn {turn.turn_index}] ({turn.role}/{turn.agent_name}): {turn.content}")
    return "\n".join(lines)


def build_extraction_messages(
    chunk: RawDialogueChunk,
    *,
    max_facts: int,
    cumulative_turn_count: int,
    carry_forward_summary: str = "(none — first chunk)",
) -> list[dict[str, str]]:
    turn_start, turn_end = chunk.turn_index_range
    user_prompt = USER_PROMPT_TEMPLATE.format(
        workflow_id=chunk.workflow_id,
        chunk_index=chunk.chunk_index,
        turn_start=turn_start,
        turn_end=turn_end,
        max_facts=max_facts,
        carry_forward_summary=carry_forward_summary,
        formatted_turns=format_turns_for_prompt(chunk),
        cumulative_turn_count=cumulative_turn_count,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
