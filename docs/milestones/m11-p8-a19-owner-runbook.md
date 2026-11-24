# M11 P8 A19 owner runbook

**A19 adopted. Dispatch remains on hold.** The owner approved the amendment and confirmed continued outcome blindness on 24 November 2026. The branch configures `final-a19`. A separate signed launch approval is still required.

## Approved A19 amendment

**Stage P8 · identity `final-a19` · decision date 2026-11-24.**

**Change.** Configure final-a19 for one governed confirmatory dispatch. Preserve the frozen estimands, scenario and seed assignments, RNG sequences, model and inference methods, simulation counts, acceptance thresholds, output semantics and fresh-last ordering. Keep isolated fits, three workers for R1/R3/REFIT and a fresh two-worker pool for each R2 pair. Retain the 4,000,000,000-byte aggregate sampled RSS ceiling and 10,800-second deadline. Require owner-pinned private history, immutable source and runtime, durable consumption before dispatch, and a protected terminal receipt followed by a within-budget certification decision. Keep outcomes private. The final, final-a14 and final-a15 identities remain permanently consumed. Failure or interruption consumes final-a19 permanently.

**Reason.** The owner confirmed continued outcome blindness after prior main-seed computation. Full-count development qualification supports this execution configuration. A19 adoption alone grants no launch authority. A separate signed owner approval is required after green CI on the exact amended release and a passing immediate host preflight.

Full-count development finished in **9,957.36 s** at **3,807,723,520 sampled bytes**, leaving **842.64 s** and **192,276,480 bytes**. Three concurrent R3 draws at scale 2.0 accounted for the peak. The 19 reduced-reference result and RNG comparisons matched exactly. Keep the existing scheduling and numerical paths. Operational headroom comes from a quiet host, stable power and leaving the execution group alone. The development run does not bound between-sample RSS or qualify the confirmatory workload.

## 1. Publish the adopted source

The adoption renderer has already applied the approved record and updated both digest pins. Start with the allowlisted staging commands below.

Owner-only Git commands from the repository root. Review diffs before committing. The tracked local continuation in `docs/evidence/m11-p8-attempts/attempts.log`, ignored history, raw outputs, private archive, runtime inputs, journals and trust files stay outside staging.

```zsh
set -euo pipefail
test "$(git branch --show-current)" = m11-p8-governance
git diff --cached --quiet
git add -- docs/milestones/m11-p8-a19-owner-runbook.md \
  scripts/{generate_m11_evidence,prepare_m11_p8_a19}.py \
  src/decision_intelligence/customer_value/{p8_candidate,p8_journal,predictive_protocol,resource_supervision}.py \
  tests/unit/customer_value/{test_p8_a14,test_p8_execution,test_p8_governance,test_predictive_protocol}.py
test "$(git diff --cached --name-only | wc -l | tr -d ' ')" = 11
git diff --cached --quiet -- docs/evidence
git diff --cached --check
git diff --cached
```

```zsh
git commit -m "Adopt reviewed M11 P8 A19 execution governance"
rc=$(git rev-parse HEAD)
git push origin HEAD:refs/heads/m11-p8-governance
ci=$(gh run list --workflow ci.yml --event push --commit "$rc" --limit 1 \
  --json databaseId --jq '.[0].databaseId // empty')
test -n "$ci"
gh run watch "$ci" --exit-status
test "$(gh run view "$ci" --json headSha,conclusion,jobs --jq \
  ".headSha == \"$rc\" and .conclusion == \"success\" and
   (.jobs | length == 4 and all(.[]; .conclusion == \"success\"))")" = true
```

Repeat the lookup if the run is still being queued. Prior CI on `9cb0416` does not cover the amended release.

## 2. Deploy the exact clean source and preflight the host

Run on the designated Mac as the owner. Reuse the existing root-controlled Python, dedicated worker, private history and append-only journal. Clone a reviewed Git bundle into a new root-controlled directory so the current deployment stays intact. Do not copy the dirty working tree.

```zsh
base=/private/var/db/m11-p8-release
trust=/private/etc/m11-p8
release="$base/source-a19-${rc[1,12]}"
python="$base/python/bin/python3.12"
worker_home="$base/worker"
private="$worker_home/Library/Application Support/M11-P8-Private"
run_dir="$private/staging/m11-p8-final-a19"
stage=$(mktemp -d /private/tmp/m11-a19-release.XXXXXXXX)
chmod 700 "$stage"
git bundle create "$stage/release.bundle" refs/heads/m11-p8-governance
sudo test ! -e "$release"
sudo /usr/bin/git clone --branch m11-p8-governance "$stage/release.bundle" "$release"
sudo chown -Rh root:wheel "$release"
sudo chmod -R go-w "$release"
test "$(sudo /usr/bin/git -C "$release" rev-parse HEAD)" = "$rc"
test -z "$(sudo /usr/bin/git -C "$release" status --porcelain --untracked-files=all)"

as_p8() {
  sudo /usr/bin/env -i HOME="$worker_home" PATH=/usr/bin:/bin LC_ALL=C TMPDIR=/private/tmp \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$python" -I -B "$@"
}
```

Use stable AC power, keep the Mac awake, stop competing workloads and check memory pressure and free space on the private staging volume. The next block repeats the privilege-boundary smoke on the amended deployment with a disposable canary. Stop before signing if it fails.

## 3. Bind and sign only with separate launch authorisation

Run this after reviewing host conditions. It checks the successor gate, exact Git/CI identity, full pinned history, empty successor staging and the live journal. The smoke uses a separate canary and leaves the live journal empty. Only an unsigned record is produced. No reservation or seed resolution occurs.

```zsh
umask 077
approval="$stage/final-a19.json"
canary=$(sudo mktemp "$base/a19-canary.XXXXXXXX")
sudo chgrp "$(sudo stat -f '%Sg' "$trust/consumption.jsonl")" "$canary"
sudo chmod 0640 "$canary"
sudo chflags sappnd "$canary"
as_p8 - "$release" "$rc" "$ci" "$canary" > "$approval" <<'PY'
import fcntl
import json
import os
import runpy
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
from decision_intelligence.customer_value import p8_candidate, p8_journal, p8_private
from decision_intelligence.customer_value import predictive_protocol as protocol

runner = runpy.run_path(str(repo / "scripts/generate_m11_evidence.py"))
protocol.require_successor_amendment()
assert not protocol.validate_protocol()
owner = p8_private._ROOT.lstat().st_uid
assert owner != 0
private = p8_private.open_private_evidence(repo, owner=owner)
attempt = p8_candidate.A19_ATTEMPT
assert private.run_dir(attempt) == private.staging / "m11-p8-final-a19"
runner["_require_private_available"](private, private.run_dir(attempt))
history = runner["verify_successor_history"](private)
live, canary = runner["_journal_path"](), Path(sys.argv[4])
with p8_journal.open_journal(live, writable=True) as journal:
    assert journal.records() == []
with p8_journal.open_journal(canary, writable=True) as journal:
    assert os.write(journal.fd, b"development canary\n") == 19
    fcntl.fcntl(journal.fd, fcntl.F_FULLFSYNC)
worker = private.root.stat()
probe = """
import os, sys
from pathlib import Path
repo, live, canary = map(Path, sys.argv[1:4])
sys.path.insert(0, str(repo / "src"))
from decision_intelligence.customer_value import p8_immutable, p8_journal
from decision_intelligence.customer_value.resource_supervision import await_supervisor
assert await_supervisor().peak_gb > 0 and os.geteuid() == int(sys.argv[4]) != 0
p8_immutable.binding(repo)
with p8_journal.open_journal(live) as journal:
    assert journal.records() == []
assert canary.read_bytes() == b"development canary\\n"
assert all(not os.access(path, os.W_OK) for path in (live, canary, repo, Path(sys.prefix)))
"""
sample = runner["supervise"](
    [sys.executable, "-I", "-B", "-c", probe, str(repo), str(live), str(canary), str(owner)],
    timeout_s=120, memory_bytes=4_000_000_000, credentials=(owner, worker.st_gid))
assert sample.status == "complete" and sample.returncode == 0 and live.stat().st_size == 0
record = p8_candidate.release_binding(
    repo, attempt=attempt, manifest=protocol.protocol_digest(),
    code=protocol.protocol_code_digest(), worker_uid=owner)
assert record["commit"] == sys.argv[2] and record["history_sha256"] == history
assert sys.argv[3].isdigit() and int(sys.argv[3]) > 0
record.update(ci_run_id=sys.argv[3], host_preflight_passed=True)
print(json.dumps(record, indent=2, sort_keys=True))
PY
```

The independent signer reviews that record against CI, the candidate digests, the measured host and the private evidence paths. After separate approval for one launch, the signer uses the existing trusted key. Set `OWNER_SIGNING_KEY` to its private path on the signing host. Return the detached signature through the owner’s private channel if signing elsewhere.

```zsh
/usr/bin/ssh-keygen -Y sign -f "$OWNER_SIGNING_KEY" -n m11-p8-final-a19 "$approval"
/usr/bin/ssh-keygen -Y verify -f "$trust/allowed_signers" -I m11-p8-owner \
  -n m11-p8-final-a19 -s "$approval.sig" < "$approval"
sudo test ! -e "$trust/final-a19.json"
sudo test ! -L "$trust/final-a19.json"
sudo test ! -e "$trust/final-a19.json.sig"
sudo test ! -L "$trust/final-a19.json.sig"
sudo install -o root -g wheel -m 0444 "$approval.sig" "$trust/final-a19.json.sig"
sudo install -o root -g wheel -m 0444 "$approval" "$trust/final-a19.json"
```

## 4. Launch once, observe metadata, then verify privately

After approval installation, with preflight still current, extract the signed manifest and code digests and dispatch exactly once from a stable local terminal. Keep the supervisor transcript in owner-only storage. A refusal or failure calls for owner review before any further action. Never clear a claim or retry a consumed identity.

```zsh
manifest=$("$python" -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$approval")
code=$("$python" -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_sha256"])' "$approval")
caffeinate -is sudo /usr/bin/env -i HOME="$worker_home" PATH=/usr/bin:/bin LC_ALL=C \
  TMPDIR=/private/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$python" -I -B \
  "$release/scripts/generate_m11_evidence.py" --mode final \
  --expect-manifest "$manifest" --expect-code "$code" \
  2>&1 | tee "$stage/supervisor.log"
```

Observe the supervisor transcript and, if needed, the ledger’s last phase and state. Do not print the ledger or inspect reports while the attempt runs.

```zsh
as_p8 - "$run_dir/ledger.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("No execution ledger yet")
else:
    ledger = json.loads(path.read_text())
    events = ledger.get("events", [])
    print(ledger["status"], [(e["phase"], e["state"]) for e in events[-1:]])
PY
```

A staged `supervision_pending` ledger or a `supervision_complete` receipt alone is insufficient. A valid `certified` journal row proves the decision followed a durable terminal receipt within the measured deadline and RSS ceiling. The decision row publishes that measurement. A crash before its publication leaves the attempt uncertified. After the supervisor exits successfully and the owner authorises evidence verification, run:

```zsh
as_p8 "$release/scripts/generate_m11_evidence.py" --verify "$run_dir"
```

Require `run verified: complete and unaltered`. This checks the protected certification, phase sequence, registry and file hashes without deserialising pickles. Review `record.md`, `summary.json` and the CSV tables only in private staging after outcome access is authorised. Use the journal certification for definitive resource measurements. Report resource figures are a pre-report snapshot. Operational completion and the frozen scientific acceptance decisions are reported separately.
