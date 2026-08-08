"""Tests for StructuredStreamParser prose-tolerant parsing and schema-name capping.

Defect: `StructuredStreamParser.feed()` accumulated the streamed buffer and
handed it to `partialjson.json_parser.JSONParser.parse`, which internally
dispatches on `buffer[0]` (`JSONParser.parse_any` -> `self.parsers.get(s[0])`).
If the first character isn't one of `{ [ " t f n <digit> <ws>`, it raises and
`feed()` swallows the exception, returning None forever. Anthropic models
streamed through the Responses API reliably prepend prose
("I need to analyze this field...\n\n{...}") before the JSON payload, so
every parse attempt failed for the whole stream.

Separately, `type_to_text_format_param(schema)` derives the OpenAI
`json_schema.name` from the Pydantic model's `__name__` with no length cap,
so models with long/generated class names trip OpenAI's 64-char limit on
`json_schema.name` and the request 400s before a single token streams back.
"""

from pydantic import BaseModel

from clinicalagent.llm import StructuredStreamParser, cap_schema_name


class SamplePrediction(BaseModel):
    decision: str
    confidence: float


def test_clean_json_from_char_zero_regression():
    """(d) Clean JSON from char 0 must keep working byte-identically (gemini path)."""
    parser = StructuredStreamParser(SamplePrediction)

    # Partial buffer mid-stream: not yet valid/complete -> no result, no exception.
    assert parser.feed('{"decision": "appro') is None

    # Completed buffer: parses immediately, same as before the fix.
    result = parser.feed('ve", "confidence": 0.95}')
    assert result == SamplePrediction(decision="approve", confidence=0.95)


def test_prose_then_json_single_chunk():
    """(a) Prose prefix followed by JSON, delivered as one chunk."""
    parser = StructuredStreamParser(SamplePrediction)
    chunk = (
        "I need to analyze this field carefully before answering.\n\n"
        '{"decision": "approve", "confidence": 0.87}'
    )
    result = parser.feed(chunk)
    assert result == SamplePrediction(decision="approve", confidence=0.87)


def test_prose_split_across_chunks_json_arrives_later():
    """(b) Prose-only buffers (no `{`/`[` yet) are 'not yet parseable', not failures."""
    parser = StructuredStreamParser(SamplePrediction)

    # Pure prose, no JSON opener anywhere yet -> not a failure, just no result.
    assert parser.feed("I need to think about this ") is None
    assert parser.feed("for a moment before responding.\n\n") is None

    # JSON opener arrives, but the object is still incomplete.
    assert parser.feed('{"decision": "reject"') is None

    # JSON completes.
    result = parser.feed(', "confidence": 0.42}')
    assert result == SamplePrediction(decision="reject", confidence=0.42)


def test_fenced_json_payload():
    """(c) Markdown code fences around the payload must be tolerated."""
    parser = StructuredStreamParser(SamplePrediction)
    chunk = (
        "Here is my answer:\n"
        "```json\n"
        '{"decision": "approve", "confidence": 0.99}\n'
        "```\n"
    )
    result = parser.feed(chunk)
    assert result == SamplePrediction(decision="approve", confidence=0.99)


def test_fenced_json_payload_split_across_chunks():
    """(c) Fenced payload where the fence markers themselves arrive in pieces."""
    parser = StructuredStreamParser(SamplePrediction)
    assert parser.feed("Sure thing.\n```json\n") is None
    assert parser.feed('{"decision": "approve"') is None
    result = parser.feed(', "confidence": 0.6}\n```')
    assert result == SamplePrediction(decision="approve", confidence=0.6)


def test_prose_containing_brace_in_quotes_before_real_payload():
    """(e) A `{` inside quoted prose text must not be locked onto as the payload start.

    The parser should attempt the spurious candidate, fail (invalid JSON /
    schema mismatch), and fall through to the next `{` candidate.
    """
    parser = StructuredStreamParser(SamplePrediction)
    chunk = (
        "The field note says {this is not json at all. "
        'Real answer: {"decision": "approve", "confidence": 0.73}'
    )
    result = parser.feed(chunk)
    assert result == SamplePrediction(decision="approve", confidence=0.73)


def test_prose_containing_balanced_but_schema_invalid_brace():
    """(e) variant: a syntactically-valid but schema-invalid `{}` inside prose
    must not be mistaken for the real payload."""
    parser = StructuredStreamParser(SamplePrediction)
    chunk = (
        'Default config is "{}" in case of doubt. '
        'Real answer: {"decision": "reject", "confidence": 0.11}'
    )
    result = parser.feed(chunk)
    assert result == SamplePrediction(decision="reject", confidence=0.11)


def test_schema_name_capped_stable_and_unique():
    """(f) Schema name >64 chars -> capped, deterministic, unique per distinct input."""
    long_name_a = "A" * 90
    long_name_b = (
        "A" * 89 + "B"
    )  # shares a 89-char prefix with long_name_a, differs at the end

    capped_a_1 = cap_schema_name(long_name_a)
    capped_a_2 = cap_schema_name(long_name_a)
    capped_b = cap_schema_name(long_name_b)

    assert len(capped_a_1) <= 64
    assert len(capped_b) <= 64
    assert capped_a_1 == capped_a_2  # stable/deterministic
    assert capped_a_1 != capped_b  # unique for distinct inputs

    # Short names pass through untouched.
    assert cap_schema_name("Short") == "Short"


def test_schema_name_sanitizes_characters_openai_rejects():
    """OpenAI requires json_schema.name to match ^[a-zA-Z0-9_-]+$.

    Pydantic's auto-generated names for parametrized generics (e.g.
    `AgentResponseThoughtful[Union[FooArgs, BarArgs]]`, the real production
    shape) contain `[`, `]`, `,`, and spaces -- illegal regardless of length.
    Confirmed live: OpenRouter/OpenAI 400s on exactly this pattern mismatch.
    """
    import re as _re

    valid_pattern = _re.compile(r"^[a-zA-Z0-9_-]+$")
    generic_name = "AgentResponseThoughtful[Union[SubmitPredictionsArgs, LookupArgs]]"

    capped = cap_schema_name(generic_name)
    assert valid_pattern.match(capped), f"still invalid for OpenAI: {capped!r}"
    assert len(capped) <= 64

    # Deterministic and distinct across two illegal names that only differ
    # after the truncation point.
    other_generic_name = (
        "AgentResponseThoughtful[Union[SubmitPredictionsArgs, OtherArgs]]"
    )
    assert cap_schema_name(generic_name) == cap_schema_name(generic_name)
    assert cap_schema_name(generic_name) != cap_schema_name(other_generic_name)

    # A short but illegal name must also be sanitized (length isn't the only trigger).
    short_illegal = "Foo[Bar]"
    capped_short = cap_schema_name(short_illegal)
    assert valid_pattern.match(capped_short), (
        f"still invalid for OpenAI: {capped_short!r}"
    )
