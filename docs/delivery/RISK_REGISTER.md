# Risk Register

| Risk | Signal | Mitigation / cut |
|---|---|---|
| Invalid or leaked benchmark | integrity assertion fails | P0 stop; repair before model work |
| Context split excludes many low-degree nodes | eligible coverage is low | report coverage and degree slices; do not claim cold-start |
| Sparse S3 becomes expensive | epoch/memory budget exceeded | sequential sparse propagation and capped structural features |
| Adaptive gate collapses | near-zero entropy across seeds | report honestly; deploy validated ablation |
| Tuning consumes product time | 24 trials or four days reached | hard stop, freeze best validation model |
| Artifact exceeds Render RAM | RSS approaches 400MB | NumPy runtime, mmap arrays, read-only SQLite, one worker |
| Render cold start harms demo | first request takes ~1 minute | wake-up UI and authoritative local Docker demo |
| 3D view becomes cluttered | low frame rate or unreadable labels | strict node/link caps, hover-only labels, freeze simulation |
| Drug metadata licensing unclear | source disallows redistribution | use public mapping or ID fallback; never block core UX |
| Public GitHub access unavailable | connector 404/write failure | make repo public, then enable GitHub App or approved CLI |
| Public model claim is overstated | “probability/validated” language appears | automated copy checks plus model/data cards |
