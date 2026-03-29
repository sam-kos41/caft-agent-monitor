#!/usr/bin/env python3
"""Extract character sprites from pixel-agents PNGs as base64.

Reads the 6 character PNG files (112x96 each) from the pixel-agents repo
and encodes them as base64 data for the web visualizer to decode client-side.

Also encodes the speech bubble JSON sprites inline.

Usage:
    python scripts/build_pixel_sprites.py [/path/to/pixel-agents]

Outputs:
    agentdiag/static/sprites.json
"""

import base64
import json
import sys
from pathlib import Path


def main():
    # Find pixel-agents directory
    if len(sys.argv) > 1:
        pixel_agents_dir = Path(sys.argv[1])
    else:
        # Try common locations
        candidates = [
            Path.home() / "pixel-agents",
            Path(__file__).resolve().parent.parent.parent / "pixel-agents",
        ]
        pixel_agents_dir = None
        for c in candidates:
            if (c / "webview-ui" / "public" / "assets" / "characters").exists():
                pixel_agents_dir = c
                break
        if pixel_agents_dir is None:
            print("Error: pixel-agents directory not found.")
            print("Usage: python scripts/build_pixel_sprites.py /path/to/pixel-agents")
            sys.exit(1)

    chars_dir = pixel_agents_dir / "webview-ui" / "public" / "assets" / "characters"
    sprites_dir = pixel_agents_dir / "webview-ui" / "src" / "office" / "sprites"

    # Encode character PNGs as base64
    characters = {}
    for png in sorted(chars_dir.glob("char_*.png")):
        characters[png.stem] = base64.b64encode(png.read_bytes()).decode()
        print(f"  Encoded {png.name} ({png.stat().st_size} bytes)")

    if not characters:
        print(f"Error: No character PNGs found in {chars_dir}")
        sys.exit(1)

    # Load speech bubble sprites (JSON pixel data)
    bubbles = {}
    for name in ["bubble-permission", "bubble-waiting"]:
        bubble_path = sprites_dir / f"{name}.json"
        if bubble_path.exists():
            bubbles[name] = json.loads(bubble_path.read_text())
            print(f"  Loaded {name}.json")

    # Build output
    output = {
        "characters": characters,
        "bubbles": bubbles,
        "meta": {
            "char_frame_w": 16,
            "char_frame_h": 32,
            "char_frames_per_row": 7,
            "directions": ["down", "up", "right"],
            "char_count": len(characters),
        },
    }

    # Write to static directory
    out_path = Path(__file__).resolve().parent.parent / "agentdiag" / "static" / "sprites.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output))
    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {len(characters)} characters + {len(bubbles)} bubbles to {out_path}")
    print(f"  Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
