import json
import re
from typing import Any, Dict, List, Union, Optional, Tuple
from enum import Enum

class ParseState(Enum):
    NORMAL = "normal"
    IN_STRING = "in_string"
    IN_KEY = "in_key"
    ESCAPE = "escape"
    IN_COMMENT_LINE = "in_comment_line"
    IN_COMMENT_BLOCK = "in_comment_block"

def from_str(input_str: str) -> Any:
    """
    Parse a string (potentially from LLM response) into JSON.
    Handles multiple fallback strategies for malformed JSON.
    
    Args:
        input_str: The raw string to parse (e.g., LLM response)
        
    Returns:
        Parsed JSON object (dict, list, or primitive)
    """
    if not input_str or not input_str.strip():
        return None
    
    # Strategy 1: Try standard JSON parsing
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    markdown_json = extract_from_markdown(input_str)
    if markdown_json:
        try:
            return json.loads(markdown_json)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find JSON objects/arrays in text
    json_objects = extract_json_objects(input_str)
    if json_objects:
        # Try to parse each found object
        for obj_str in json_objects:
            try:
                return json.loads(obj_str)
            except json.JSONDecodeError:
                # Try fixing malformed JSON
                fixed = fix_malformed_json(obj_str)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue
    
    # Strategy 4: Fix malformed JSON on entire input
    fixed = fix_malformed_json(input_str.strip())
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Return as string if all else fails
    return input_str.strip()

def extract_from_markdown(text: str) -> Optional[str]:
    """Extract JSON from markdown code blocks."""
    # Look for ```json blocks
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def extract_json_objects(text: str) -> List[str]:
    """Extract potential JSON objects or arrays from text."""
    results = []
    
    # Find all potential JSON starts
    for i, char in enumerate(text):
        if char in ['{', '[']:
            # Try to find matching closing bracket
            end_idx = find_matching_bracket(text, i)
            if end_idx != -1:
                potential_json = text[i:end_idx + 1]
                results.append(potential_json)
    
    return results

def find_matching_bracket(text: str, start_idx: int) -> int:
    """Find the matching closing bracket for a JSON object/array."""
    if start_idx >= len(text):
        return -1
    
    open_char = text[start_idx]
    close_char = '}' if open_char == '{' else ']'
    
    count = 1
    in_string = False
    escape = False
    
    for i in range(start_idx + 1, len(text)):
        char = text[i]
        
        if escape:
            escape = False
            continue
            
        if char == '\\':
            escape = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1
                if count == 0:
                    return i
    
    return -1

def fix_malformed_json(text: str) -> str:
    """
    Fix common JSON formatting issues in LLM responses.
    Handles unquoted strings, single quotes, trailing commas, etc.
    """
    # Remove comments
    text = remove_comments(text)
    
    # State machine for parsing
    result = []
    state = ParseState.NORMAL
    buffer = ""
    i = 0
    bracket_stack = []
    
    while i < len(text):
        char = text[i]
        
        if state == ParseState.ESCAPE:
            result.append(char)
            state = ParseState.IN_STRING
            i += 1
            continue
        
        if state == ParseState.IN_STRING:
            if char == '\\':
                result.append(char)
                state = ParseState.ESCAPE
            elif char == '"':
                result.append(char)
                state = ParseState.NORMAL
            else:
                result.append(char)
            i += 1
            continue
        
        # Normal state
        if char == '"':
            result.append(char)
            state = ParseState.IN_STRING
        elif char == "'":
            # Convert single quotes to double quotes
            result.append('"')
            i += 1
            # Find closing single quote
            while i < len(text) and text[i] != "'":
                if text[i] == '"':
                    result.append('\\"')
                else:
                    result.append(text[i])
                i += 1
            result.append('"')
        elif char in ['{', '[']:
            result.append(char)
            bracket_stack.append(char)
        elif char in ['}', ']']:
            # Remove trailing comma if present
            j = len(result) - 1
            while j >= 0 and result[j] in [' ', '\n', '\t']:
                j -= 1
            if j >= 0 and result[j] == ',':
                result = result[:j] + result[j+1:]
            result.append(char)
            if bracket_stack and ((char == '}' and bracket_stack[-1] == '{') or 
                                 (char == ']' and bracket_stack[-1] == '[')):
                bracket_stack.pop()
        elif char == ':':
            # Check if we need to quote the key
            j = len(result) - 1
            while j >= 0 and result[j] in [' ', '\n', '\t']:
                j -= 1
            
            # If the previous non-whitespace char is not a quote, we need to quote the key
            if j >= 0 and result[j] != '"':
                # Find the start of the key
                key_end = j + 1
                while j >= 0 and result[j] not in [',', '{', '[', '\n']:
                    j -= 1
                key_start = j + 1
                
                # Extract and quote the key
                key = ''.join(result[key_start:key_end]).strip()
                result = result[:key_start] + ['"', key, '"'] + result[key_end:]
            
            result.append(char)
        else:
            result.append(char)
        
        i += 1
    
    # Close any unclosed brackets
    while bracket_stack:
        bracket = bracket_stack.pop()
        if bracket == '{':
            result.append('}')
        else:
            result.append(']')
    
    return ''.join(result)

def remove_comments(text: str) -> str:
    """Remove both line and block comments from text."""
    # Remove line comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    # Remove block comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

# Example usage and test cases
if __name__ == "__main__":
    # Test cases demonstrating the parser's capabilities
    test_cases = [
        # Standard JSON
        '{"name": "John", "age": 30}',
        
        # JSON in markdown
        '''Here's the result:
        ```json
        {"status": "success", "data": [1, 2, 3]}
        ```
        ''',
        
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
        '''{
            // User information
            "name": "John", // Full name
            /* Age in years */
            "age": 30
        }''',
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}:")
        print(f"Input: {test}")
        result = from_str(test)
        print(f"Output: {result}")
        print(f"Type: {type(result)}")