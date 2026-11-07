# CB reels: architecture

One coloring page goes in, one 1080x1920 reel comes out, with no human in the loop. A *job* (`engine/job.py`) carries the page through the stages below. Every stage writes its status to `state.json`, which the app window draws, and appends events to `log.jsonl`, so a restarted job resumes without paying twice for the same clip.

```
page -> colors -> strokes -> ink -> rough -> courier -> spot -> judge -> sound -> cut -> final
```

| Stage | What it does |
|---|---|
| page | Reads the line art and splits it into fillable regions. |
| colors | Picks a marker for every region, from a colored target image or a planned palette. |
| strokes | Plans every marker movement as one stroke timeline, replaying gestures measured on reference videos of real marker coloring. |
| ink | Digital paper: alcohol ink modelled as a wet film that levels out, so stroke lines melt away like real markers. |
| rough | For each shot, renders the page coloring itself plus a stand-in hand on the nib. |
| courier | Sends rough clips to a video model and fetches the results, with a cost estimate and budget check per clip. |
| spot | Spot the difference: the result may only change near the hand. Ink can't vanish, appear early or change color. |
| judge | Optional vision check that the hand looks real (no extra fingers, no floating nib). |
| sound | Soft marker-on-paper sound on every stroke. No music. |
| cut | Edits the shots into one reel: jump cuts, light grade, grain, handheld shake. |
| final | Checks the file is postable: format, length, audio, no black or frozen frames, ends on the finished page. |

Each hand shot has a retry ladder: model 1, checks, model 2, checks, then a fallback still shot.

## In this repo so far
- `make_reel.py`: command-line entry point.
- `engine/job.py`: the job state machine and append-only log.
- `engine/paths.py`: where data and user files live (source or packaged app).
- `engine/video.py`: ffmpeg helpers for writing and reading frames.
- `engine/final.py`: the final postability check.

The other stages are being cleaned up and will follow.
