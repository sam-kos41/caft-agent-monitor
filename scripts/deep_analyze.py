#!/usr/bin/env python3
import json
import sys

def deep_analyze(jsonl_path):
    """Deep analysis of trace for annotation purposes."""
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]

    # Find first user message with actual content
    first_user_content = None
    for l in lines[:20]:
        if l.get('type') == 'user':
            msg = l.get('message', {}).get('content', '')
            if isinstance(msg, str) and len(msg) > 20:
                first_user_content = msg[:500]
                break
            elif isinstance(msg, list) and len(msg) > 0:
                for item in msg:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        first_user_content = item.get('text', '')[:500]
                        break
                if first_user_content:
                    break

    # Find last 5 messages
    last_messages = []
    for l in lines[-10:]:
        msg_type = l.get('type')
        if msg_type in ['user', 'assistant']:
            content = l.get('message', {}).get('content', '')
            if isinstance(content, str):
                last_messages.append({'type': msg_type, 'content': content[:300]})
            elif isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
                if text_parts:
                    last_messages.append({'type': msg_type, 'content': text_parts[0][:300]})

    # Count error types
    error_messages = []
    for l in lines:
        if l.get('type') == 'progress':
            msg = l.get('data', {}).get('message', {}).get('message', {})
            content_items = msg.get('content', [])
            if isinstance(content_items, list):
                for item in content_items:
                    if isinstance(item, dict) and item.get('is_error'):
                        error_messages.append(item.get('content', '')[:200])

    return {
        'first_user_content': first_user_content,
        'last_messages': last_messages,
        'error_messages': error_messages[:10],  # First 10 errors
        'total_errors': len(error_messages)
    }

if __name__ == '__main__':
    result = deep_analyze(sys.argv[1])
    print(json.dumps(result, indent=2))
