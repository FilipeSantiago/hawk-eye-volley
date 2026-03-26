## Lazy Export Format (v2)

The ball detection pipeline supports a lazy-load export designed for Electron/Node.
Use `--export-format lazy` (default) for streaming access, or `--export-format monolithic` for the legacy single JSON.

Sample output folder structure:
- `data/processing_folder/ball_predictions/<video_stem>/session.meta.json`
- `data/processing_folder/ball_predictions/<video_stem>/frames.ndjson`
- `data/processing_folder/ball_predictions/<video_stem>/frames.index.json`

The legacy monolithic export writes:
- `data/processing_folder/ball_predictions/<video_stem>.json`

### Reading a frame by index

```python
import json
from pathlib import Path

index = json.loads(Path("frames.index.json").read_text(encoding="utf-8"))
target = 123
item = next(entry for entry in index["items"] if entry["frame_index"] == target)
with open("frames.ndjson", "rb") as handle:
    handle.seek(item["offset"])
    line = handle.read(item["length"])
frame = json.loads(line)
print(frame["frame"], frame["index"], len(frame["candidates"]))
```
