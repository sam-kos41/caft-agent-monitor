#!/usr/bin/env python3
"""Extract key information from trace JSONL files for annotation."""

import json
import sys
import os

def extract_trace_info(jsonl_path, session_id, detections):
    """Extract key information from a trace JSONL file."""
    events = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                continue

    # Parse events to extract structured info
    parsed = []
    for i, event in enumerate(events):
        info = {
            'line': i,
            'type': event.get('type', 'unknown'),
        }

        # Get message content
        if event.get('type') == 'user':
            msg = event.get('message', '')
            if isinstance(msg, dict):
                msg = msg.get('content', '')
            if isinstance(msg, list):
                texts = []
                for part in msg:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        texts.append(part.get('text', '')[:200])
                msg = ' '.join(texts)
            info['content'] = str(msg)[:300]

        elif event.get('type') == 'assistant':
            msg = event.get('message', {})
            if isinstance(msg, dict):
                content = msg.get('content', [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get('type') == 'tool_use':
                                tool_name = part.get('name', '')
                                tool_input = part.get('input', {})
                                # Summarize tool input
                                if tool_name in ('Read', 'read_file'):
                                    inp = tool_input.get('file_path', tool_input.get('path', ''))
                                    info.setdefault('tools', []).append(f"{tool_name}({inp})")
                                elif tool_name in ('Write', 'write_file'):
                                    inp = tool_input.get('file_path', tool_input.get('path', ''))
                                    info.setdefault('tools', []).append(f"{tool_name}({inp})")
                                elif tool_name in ('Edit', 'edit_file'):
                                    inp = tool_input.get('file_path', tool_input.get('path', ''))
                                    info.setdefault('tools', []).append(f"{tool_name}({inp})")
                                elif tool_name in ('Bash', 'bash'):
                                    cmd = tool_input.get('command', '')[:100]
                                    info.setdefault('tools', []).append(f"Bash({cmd})")
                                elif tool_name in ('Grep', 'grep'):
                                    pat = tool_input.get('pattern', '')[:50]
                                    info.setdefault('tools', []).append(f"Grep({pat})")
                                elif tool_name in ('Glob', 'glob'):
                                    pat = tool_input.get('pattern', '')[:50]
                                    info.setdefault('tools', []).append(f"Glob({pat})")
                                elif tool_name == 'Task':
                                    prompt = tool_input.get('prompt', '')[:80]
                                    info.setdefault('tools', []).append(f"Task({prompt})")
                                elif tool_name == 'TaskCreate':
                                    prompt = tool_input.get('prompt', '')[:80]
                                    info.setdefault('tools', []).append(f"TaskCreate({prompt})")
                                elif tool_name == 'TaskUpdate':
                                    info.setdefault('tools', []).append(f"TaskUpdate")
                                else:
                                    info.setdefault('tools', []).append(tool_name)
                            elif part.get('type') == 'text':
                                text = part.get('text', '')[:200]
                                info['text'] = text
            info['content'] = info.get('text', '')[:200]

        elif event.get('type') == 'tool_result':
            result = event.get('result', '')
            is_error = event.get('is_error', False)
            if isinstance(result, str):
                info['content'] = result[:200]
            info['is_error'] = is_error

        parsed.append(info)

    # Build summary
    result = {
        'session_id': session_id,
        'total_events': len(events),
        'event_types': {},
    }

    for p in parsed:
        t = p['type']
        result['event_types'][t] = result['event_types'].get(t, 0) + 1

    # Get user messages
    user_msgs = []
    for p in parsed:
        if p['type'] == 'user' and p.get('content'):
            user_msgs.append(p['content'][:200])
    result['user_messages'] = user_msgs

    # Get tool sequence
    tool_seq = []
    for p in parsed:
        if p.get('tools'):
            tool_seq.extend(p['tools'])
    result['tool_sequence'] = tool_seq

    # Get errors
    errors = []
    for p in parsed:
        if p.get('is_error'):
            errors.append({'line': p['line'], 'content': p.get('content', '')[:200]})
    result['errors'] = errors

    # Get assistant text snippets (first and last)
    assistant_texts = []
    for p in parsed:
        if p['type'] == 'assistant' and p.get('content'):
            assistant_texts.append(p['content'][:200])
    if assistant_texts:
        result['first_assistant'] = assistant_texts[0][:200]
        result['last_assistant'] = assistant_texts[-1][:200]

    # Check for detection-relevant patterns
    # Re-read patterns (context_loss)
    read_files = {}
    for i, p in enumerate(parsed):
        if p.get('tools'):
            for tool in p['tools']:
                if tool.startswith('Read('):
                    path = tool[5:-1]
                    if path not in read_files:
                        read_files[path] = []
                    read_files[path].append(i)

    re_reads = {k: v for k, v in read_files.items() if len(v) > 1}
    result['re_reads'] = {k: v for k, v in list(re_reads.items())[:5]}

    # Check for consecutive same-tool calls (step_repetition)
    consecutive_same = 0
    max_consecutive = 0
    prev_tool = None
    for t in tool_seq:
        if t == prev_tool:
            consecutive_same += 1
            max_consecutive = max(max_consecutive, consecutive_same)
        else:
            consecutive_same = 1
        prev_tool = t
    result['max_consecutive_same_tool'] = max_consecutive

    # Check for long read-only sequences (tool_thrashing)
    read_only_tools = {'Read', 'Grep', 'Glob', 'Task', 'read_file', 'grep', 'glob'}
    max_read_only_streak = 0
    current_streak = 0
    for t in tool_seq:
        tool_name = t.split('(')[0]
        if tool_name in read_only_tools:
            current_streak += 1
            max_read_only_streak = max(max_read_only_streak, current_streak)
        else:
            current_streak = 0
    result['max_read_only_streak'] = max_read_only_streak

    return result


def main():
    # Load selection and candidates
    base_dir = '/Users/samkoscelny/GazeVLM-local/agentdiag'

    with open(os.path.join(base_dir, 'annotation_selection.json')) as f:
        selection = json.load(f)

    with open(os.path.join(base_dir, 'annotation_candidates.json')) as f:
        candidates = json.load(f)

    # Create session_id -> path mapping
    session_paths = {s['session_id']: s['path'] for s in selection}

    # Process each trace
    all_results = []
    for cand in candidates:
        idx = cand['idx']
        sid = cand['session_id']
        path = session_paths.get(sid, '')

        if not path or not os.path.exists(path):
            print(f"SKIP idx={idx}: path not found: {path}", file=sys.stderr)
            continue

        print(f"Processing idx={idx} sid={sid[:8]}... ({cand['event_count']} events)", file=sys.stderr)

        info = extract_trace_info(path, sid, cand.get('detections', []))
        info['idx'] = idx
        info['detections'] = cand.get('detections', [])
        info['user_goal'] = cand.get('user_goal', '')[:300]
        all_results.append(info)

    # Output
    print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
