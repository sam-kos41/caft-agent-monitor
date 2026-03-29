#!/usr/bin/env python3
"""
Full trace analyzer for CAFT annotation.
Extracts: user goal, completion status, and potential failures.
"""
import json
import sys

def analyze_full_trace(jsonl_path):
    """Comprehensive trace analysis."""
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]

    # Extract all user messages
    user_messages = []
    for i, l in enumerate(lines):
        if l.get('type') == 'user':
            msg = l.get('message', {}).get('content', '')
            if isinstance(msg, str) and len(msg) > 10:
                user_messages.append((i, msg))
            elif isinstance(msg, list):
                text_parts = [item.get('text', '') for item in msg if isinstance(item, dict) and item.get('type') == 'text']
                if text_parts and len(text_parts[0]) > 10:
                    user_messages.append((i, text_parts[0]))

    # Extract all assistant messages
    assistant_messages = []
    for i, l in enumerate(lines):
        if l.get('type') == 'assistant':
            msg = l.get('message', {}).get('content', '')
            if isinstance(msg, str):
                assistant_messages.append((i, msg))
            elif isinstance(msg, list):
                text_parts = [item.get('text', '') for item in msg if isinstance(item, dict) and item.get('type') == 'text']
                if text_parts:
                    assistant_messages.append((i, text_parts[0]))

    # Track tool calls and errors
    tool_calls = []
    errors = []
    for i, l in enumerate(lines):
        if l.get('type') == 'progress':
            msg = l.get('data', {}).get('message', {}).get('message', {})
            content_items = msg.get('content', [])
            if isinstance(content_items, list):
                for item in content_items:
                    if isinstance(item, dict):
                        if item.get('is_error'):
                            errors.append((i, item.get('content', '')[:300]))
                        if item.get('type') == 'tool_use':
                            tool_name = item.get('name', '')
                            tool_calls.append((i, tool_name))

    # Analyze completion
    last_user_idx = user_messages[-1][0] if user_messages else 0
    last_assistant_idx = assistant_messages[-1][0] if assistant_messages else 0

    # Did agent respond after last user message?
    agent_completed = last_assistant_idx > last_user_idx

    # Check for premature termination indicators
    last_user_msg = user_messages[-1][1] if user_messages else ""
    last_assistant_msg = assistant_messages[-1][1] if assistant_messages else ""

    # Pattern detection
    repeated_tools = []
    tool_sequence = [t[1] for t in tool_calls]
    for i in range(len(tool_sequence) - 4):
        if tool_sequence[i:i+3] == tool_sequence[i+2:i+5]:
            repeated_tools.append((i, tool_sequence[i:i+3]))

    return {
        'total_lines': len(lines),
        'user_message_count': len(user_messages),
        'assistant_message_count': len(assistant_messages),
        'first_user_message': user_messages[0][1][:500] if user_messages else "",
        'last_user_message': last_user_msg[:500],
        'last_assistant_message': last_assistant_msg[:500],
        'agent_completed': agent_completed,
        'total_errors': len(errors),
        'error_samples': errors[:5],
        'tool_call_count': len(tool_calls),
        'repeated_tool_patterns': repeated_tools[:3],
        'final_exchange': {
            'last_3_user': [msg[1][:200] for msg in user_messages[-3:]],
            'last_3_assistant': [msg[1][:200] for msg in assistant_messages[-3:]]
        }
    }

if __name__ == '__main__':
    result = analyze_full_trace(sys.argv[1])
    print(json.dumps(result, indent=2))
