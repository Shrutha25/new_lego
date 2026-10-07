# Per-step glTF models

One model per assembly step, referenced from `states.json`:

| Step | File | Buffer it needs (must sit alongside it) |
|---|---|---|
| 1 - Chassis shown | `step_1.gltf` | `Chassis.bin` |
| 2a - Right side wheels | `step_2a.gltf` | `Tires Right.bin` |
| 2b - Left side wheels | `step_2b.gltf` | `Tires Left.bin` |
| 3 - Windshield | `step_3.gltf` | `Windshield.bin` |
| 4 - Roof | `step_4.gltf` | `Roof.bin` |
| Shown once everything is confirmed | `Car.gltf` (`config.FINAL_MODEL`) | `Car.bin` |

These are non-binary `.gltf` files - the `.gltf` itself is just the scene
JSON, the actual mesh data lives in the separate `.bin` file it points to.
The buffer filename has to match **exactly** what the `.gltf` references
internally (check its `buffers[].uri` field if a model won't load -
`web_server.py` 404s on any mismatch). If you'd rather manage one file
instead of a pair, export as a single `.glb` and update the matching
`"model"` path in `../../states.json` (or `config.FINAL_MODEL`) to end in
`.glb`.

To add a new step later, add a `"model"` path to that step's entry in
`../../states.json` and drop the matching file(s) here.
