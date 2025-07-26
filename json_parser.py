
import json
import re
from enum import Enum
from typing import Any

class ParseState(Enum):
    NORMAL = "normal"
    IN_STRING = "in_string"
    ESCAPE = "escape"

def from_str(input_str: str) -> Any:
    """
    Parse a string (potentially from an LLM response) into JSON.
    Heuristics:
      - Parse normal JSON
      - Extract/parse ```...``` blocks
      - Find/parse all {...}/[...] substrings
      - Fix malformed JSON (comments, single quotes, unquoted keys/values,
        numbers starting with '.', trailing/leading commas, balance braces/brackets)
      - If the whole payload is a quoted JSON blob, unquote+unescape then parse
      - Quote complex expression-like values (e.g., 314.97 + 1.32)
    """
    if not input_str or not input_str.strip():
        return None
    raw = input_str.strip()

    # Strategy 0: whole payload is quoted -> unquote + unescape + parse
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        inner = raw[1:-1]
        try:
            inner = bytes(inner, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        inner = inner.strip()
        if inner.startswith("{") or inner.startswith("["):
            fixed_inner = _fix_malformed_json(inner)
            try:
                return json.loads(fixed_inner)
            except json.JSONDecodeError:
                pass  # fall through

    # Strategy 1: Try straightforward JSON parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract and parse all code-block JSON (```json ... ```)
    md_pattern = r"```(?:\w*\s*)?\n(.*?)\n```"
    md_blocks = re.findall(md_pattern, raw, flags=re.DOTALL)
    md_results = []
    for blk in md_blocks:
        blk = blk.strip()
        try:
            md_results.append(json.loads(blk))
            continue
        except json.JSONDecodeError:
            fixed_blk = _fix_malformed_json(blk)
            try:
                md_results.append(json.loads(fixed_blk))
            except json.JSONDecodeError:
                pass
    if md_results:
        return md_results[0] if len(md_results) == 1 else md_results

    # Strategy 3: Find and parse all JSON objects/arrays in the text
    objs = _extract_json_objects(raw)
    obj_results = []
    for s in objs:
        try:
            obj_results.append(json.loads(s))
            continue
        except json.JSONDecodeError:
            fixed = _fix_malformed_json(s)
            try:
                obj_results.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    if obj_results:
        return obj_results[0] if len(obj_results) == 1 else obj_results

    # Strategy 4: Fix malformed JSON on the entire input
    fixed = _fix_malformed_json(raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return raw  # last resort: return original string

def _extract_json_objects(text: str) -> list[str]:
    """Find all balanced {...} and [...] substrings in text."""
    results = []
    stack = []
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if stack:
                open_br = stack.pop()
                if (open_br == '{' and ch == '}') or (open_br == '[' and ch == ']'):
                    if not stack and start is not None:
                        results.append(text[start:i+1])
                        start = None
                else:
                    stack.clear()
                    start = None
    if stack and start is not None:
        results.append(text[start:])
    return results

def _fix_malformed_json(text: str) -> str:
    """
    Repair common JSON issues:
      - remove comments
      - Python True/False/None -> JSON true/false/null
      - numbers starting with '.'
      - single quotes -> double quotes
      - quote unquoted keys and bareword values (but not literals/numbers)
      - quote complex unquoted expressions (e.g., '314.97 + 1.32')
      - remove trailing/leading commas
      - balance braces/brackets with a stack
    """
    # 1) Remove comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 2) Python literals
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # 3) Numbers starting with '.'
    text = re.sub(r"(?<=[:\s\[,])\.(\d+)", r"0.\1", text)

    # 4) Single-quoted strings -> double-quoted
    def _convert_single(m: re.Match) -> str:
        inner = m.group(1).replace('"', '\\"')
        return '"' + inner + '"'
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", _convert_single, text)

    # 5) Quote unquoted keys
    key_pattern = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
    text = key_pattern.sub(r'\1"\2"\3', text)

    # 6) Quote unquoted bareword values (not literals or numbers)
    def _quote_bareword_value(m: re.Match) -> str:
        prefix, value = m.group(1), m.group(2)
        if value in {"true", "false", "null"} or re.fullmatch(r"-?\d+(\.\d+)?([eE][+\-]?\d+)?", value):
            return prefix + value
        return f'{prefix}"{value}"'
    val_pattern = re.compile(r'(:\s*)([A-Za-z_][A-Za-z0-9_]*)(?=[,\}\]])')
    text = val_pattern.sub(_quote_bareword_value, text)

    # 6b) Quote complex unquoted expressions appearing as values
    def _quote_expression(m: re.Match) -> str:
        prefix, value = m.group(1), m.group(2)
        v = value.strip()
        # skip if already quoted/object/array
        if v.startswith(('"', "'", "{", "[")):
            return prefix + value
        # JSON literals or numbers -> leave
        if v in {"true", "false", "null"} or re.fullmatch(r"-?\d+(\.\d+)?([eE][+\-]?\d+)?", v):
            return prefix + v
        v = v.replace('"', '\\"')
        return f'{prefix}"{v}"'
    expr_pattern = re.compile(r'(:\s*)([^,\}\]]+)(?=\s*[,\}\]])')
    text = expr_pattern.sub(_quote_expression, text)

    # 7) Remove dangling commas
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    text = re.sub(r"([\{\[])\s*,\s*", r"\1", text)

    # 8) Balance brackets with a stack
    stack = []
    for ch in text:
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch == '}' and stack and stack[-1] == '}':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == ']':
            stack.pop()
    text += ''.join(reversed(stack))
    return text


# Example usage and test cases
if __name__ == "__main__":
    # Test cases demonstrating the parser's capabilities
    test_cases = [
        # Standard JSON
        '{"name": "John", "age": 30}',

        # JSON in markdown
        """Here's the result:
        ```json
        {"status": "success", "data": [1, 2, 3]}
        ```
        """,

        # Malformed JSON with single quotes
        "{'name': 'John', 'age': 30}",

        # Unquoted keys
        "{name: 'John', age: 30, active: true}",

        # Trailing commas
        '{"items": [1, 2, 3,], "total": 3,}',

        # Mixed content with JSON
        "The user data is {name: John, scores: [95, 87, 92]} as shown above.",

        # Incomplete JSON (streaming)
        '{"status": "processing", "items": [1, 2',

        # Comments in JSON
        """{
            // User information
            "name": "John", // Full name
            /* Age in years */
            "age": 30
        }""",

        """
        "{\n  \"intent\": \"divide\",\n  \"reasoning\": \"Now I need to perform the division.\",\n  \"numerator\": 314.97 + 1.32,\n  \"denominator\": 15.152\n}"
        
        """,
        # Markdown block with non-"json" tag
        """```test json
        {"a": 1, "b": 2}
        ```""",

        # Leading comma in array
        '{"a": [,1,2,3]}',

        # Numbers starting with a dot
        '{"a": .5, "b": [.1, .2]}',

        # Python True/False/None
        '{"ok": True, "flag": False, "data": None}',

        # Unquoted values with hyphen and slash (date/path)
        '{"date": 2024-01-02, "path": C:/Users/John}',

        # Scientific notation numbers
        '{"eps": 1e-3, "nums": [2E3, -3.5e+2]}',

        # Nested trailing commas (object + array)
        '{"list":[{"a":1}, {"b":2},],}',

        # Garbage before/after JSON + comments + trailing comma
        "blah { a: 1, /*c*/ b: 2, } trailing",

        # Unterminated array with nested objects (streaming)
        '{"a":[{"x":1}, {"y":2}',

        # Whole payload quoted with single quotes
        "'{\"city\": \"MTL\", \"ok\": true}'",

        # Uppercase bareword booleans -> should become strings
        '{"active": TRUE, "off": FALSE}',

        # URL and email as unquoted values -> should be quoted strings
        '{"url": https://example.com, "email": test@example.com}',

        # Expression value with parentheses -> should be quoted
        '{"expr": (1+2)}',

        # Markdown block with extra spaces in tag
        """```json   
        {"alpha": 1, "beta": 2}
        ```""",

        # Mixed content; first balanced object should be parsed
        "Intro text... {name: Alice, score: 42, done: false}, more text",

        # Array value with leading comma inside nested structure
        '{"outer": {"arr": [,10,20]}}',

    ]

    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}:")
        print(f"Input: {test}")
        result = from_str(test)
        print(f"Output: {result}")
        print(f"Type: {type(result)}")
        assert isinstance(result, dict), (
            f"Test {i + 1} failed: expected dict but got {type(result)}. Parsed value: {result}"
        )
        print("Type assertion passed (dict).")

