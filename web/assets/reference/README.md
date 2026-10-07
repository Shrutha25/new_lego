# Per-step reference media (optional)

The trainer screen's hint bar looks for a reference image at:

```
web/assets/reference/step_<step-id>_correct.jpg
```

e.g. `step_1_correct.jpg`, `step_2a_correct.jpg`. If it's not there, the
hint bar just says so - that's expected until you add these.

"View Reference" also opens a video player (`web/index.html`'s `#hint-modal`)
that currently has no `src` wired up. If you have short how-to-do-this-step
clips, drop them here too (e.g. `step_1_correct.mp4`) and tell Claude - it's
a one-line change in `app.js` to point the modal's `<video>` at the right
file per step.
