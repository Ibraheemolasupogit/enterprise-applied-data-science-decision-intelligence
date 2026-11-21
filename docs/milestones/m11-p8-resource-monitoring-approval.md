# M11 P8 — Resource monitoring owner approval

**Decision date:** 21 November 2026

**Repository baseline:** `d5a14a4`

The owner approved measured aggregate peak RSS with fail-closed supervision as the
acceptance method for the existing **4 GB (4,000,000,000 bytes)** memory budget. The owner
acknowledged that sampling can miss instantaneous memory spikes. The **180-minute
elapsed-time deadline** continues through final report writing.

This decision clarifies how the existing memory budget is measured. The frozen statistical
specifications and resource limits remain unchanged. The decision requires no protocol
amendment or digest repinning.

The [macOS supervisor](../../src/decision_intelligence/customer_value/resource_supervision.py)
was committed in `e580871`. Its
[development tests](../../tests/unit/customer_value/test_p8_resource_supervision.py) cover
termination on a measured memory breach, loss of monitoring, and timeouts during
serialization and report writing. Baseline CI run `44894464572` passed. The
[owner-approved P8 dry run](../evidence/m11-p8-dry-run/record.md#1-performance) reported
2.31 GB peak aggregate memory. The
[three-worker development benchmark](m11-p8-a15-draft.md) reported 2.282 GB sampled peak
aggregate RSS.

**The A15 execution hold remains in force.** The historical `final` and `final-a14`
reservations remain permanently consumed. Creating a `final-a15` reservation, clearing the
hold, and executing another final experiment require separate, explicit owner authorisation.
