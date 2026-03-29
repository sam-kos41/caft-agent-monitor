#!/usr/bin/env python3
import json
import sys

def analyze_trace(jsonl_path):
    """Extract key information from a trace JSONL file."""
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]

    # Extract first user message
    user_messages = [l for l in lines if l.get('type') == 'user']
    first_user = user_messages[0]['message']['content'] if user_messages else "No user message"

    # Count errors
    error_count = sum(1 for l in lines if l.get('type') == 'progress' and
                      l.get('data', {}).get('message', {}).get('message', {}).get('content', [{}])[0].get('is_error'))

    # Get last message
    last_msg = lines[-1] if lines else {}
    last_type = last_msg.get('type', 'unknown')

    # Extract assistant messages (responses)
    assistant_msgs = [l for l in lines if l.get('type') == 'assistant']

    return {
        'total_lines': len(lines),
        'first_user_msg': first_user[:200],
        'error_count': error_count,
        'last_msg_type': last_type,
        'assistant_msg_count': len(assistant_msgs),
        'user_msg_count': len(user_messages)
    }

if __name__ == '__main__':
    result = analyze_trace(sys.argv[1])
    print(json.dumps(result, indent=2))
