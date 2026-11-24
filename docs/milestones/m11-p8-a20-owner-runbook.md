# M11 P8 A20 owner runbook

**Hold for independent owner approval.** The configured A19 identity is consumed. A20 is proposed and has no launch authority.

## Proposed A20 amendment

**Stage P8 · proposed identity `final-a20` · owner date and outcome-blindness confirmation pending.**

**Change.** Configure final-a20 for at most one independently authorised confirmatory dispatch. Keep final, final-a14, final-a15 and final-a19 permanently consumed. Preserve the protected A19 startup-failure record and bind approval to its journal chain tip. Require the dedicated worker to verify the immutable checkout, including the Git index, the signed release and the private history under native supervision before consumption. Preserve all frozen estimands, numerical and inference methods, seeds, RNG sequences, simulation counts and scales, acceptance thresholds, reporting semantics and fresh-last ordering. Keep the qualified hybrid schedule and the 4,000,000,000-byte aggregate sampled RSS and 10,800-second limits.

**Reason.** The final-a19 attempt on release 493fd0b was consumed by a child failure with exit code 2 after 1.657 seconds. The dedicated worker could not read the root-owned Git index and stopped before an execution ledger was created. The index permissions were repaired and worker Git status and complete signed-release verification passed. The owner confirmed continued outcome blindness. A20 adoption grants no launch authority. A separate attempt-specific signed approval requires green CI on the exact amended release and a passing immediate host preflight.

## 1. Owner adoption, publication and exact-release CI

After independently approving the text, record the actual decision date and continued outcome blindness. From the repository root, set `OWNER_DECISION_DATE` and render the patch. It updates the identity, amendment list, timing list and date-bound digest pins. Review before applying. Preserve the local A19 runbook edits and the historical `attempts.log` continuation outside this commit.

```zsh
set -euo pipefail
umask 077
test "$(git branch --show-current)" = m11-p8-governance
git diff --cached --quiet
patch=$(mktemp /private/tmp/m11-a20-adoption.XXXXXXXX)
PYTHONPATH=src python3.12 -B -m scripts.prepare_m11_p8_a20 \
  --decision-date "${OWNER_DECISION_DATE:?set the independently approved date}" > "$patch"
git apply --check "$patch"
cat "$patch"
# Apply only after reviewing the approved amendment.
git apply "$patch"
git add -- docs/milestones/m11-p8-a20-owner-runbook.md \
  scripts/{generate_m11_evidence,prepare_m11_p8_a19,prepare_m11_p8_a20}.py \
  src/decision_intelligence/customer_value/{p8_candidate,p8_immutable,predictive_protocol}.py \
  tests/unit/customer_value/{test_p8_a14,test_p8_governance,test_p8_release_candidate,test_predictive_protocol}.py
git diff --cached --quiet -- docs/evidence docs/milestones/m11-p8-a19-owner-runbook.md
git diff --cached --check
git diff --cached
```

Manually publish the reviewed diff. Repeat the run lookup if CI is still being queued.

```zsh
git commit -m "Adopt reviewed M11 P8 A20 execution governance"
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

## 2. Deploy and preflight the designated Mac

Reuse the existing protected runtime, worker, private history and journal. Deploy only the exact committed source into a new root-owned checkout. Leave the A19 deployment intact. Check stable AC power, wakefulness, free private storage and memory pressure. Stop on any failure.

```zsh
base=/private/var/db/m11-p8-release
trust=/private/etc/m11-p8
release="$base/source-a20-${rc[1,12]}"
python="$base/python/bin/python3.12"
worker_home="$base/worker"
stage=$(mktemp -d /private/tmp/m11-a20-release.XXXXXXXX)
git bundle create "$stage/release.bundle" refs/heads/m11-p8-governance
sudo -v
sudo test ! -e "$release"
sudo test ! -L "$release"
sudo /usr/bin/git clone --branch m11-p8-governance "$stage/release.bundle" "$release"
sudo chown -Rh root:wheel "$release"
sudo chgrp -Rh _m11p8 "$release"
sudo chmod -R -P g+rX,g-w,o-rwx "$release"
test "$(sudo /usr/bin/git -C "$release" rev-parse HEAD)" = "$rc"

# Check read access across the checkout from the worker account, including Git metadata.
sudo -u _m11p8 -g _m11p8 /usr/bin/find "$release" \
  -exec /bin/sh -c '
    for path do
      test -r "$path" && test ! -w "$path" || exit 1
      if test -d "$path"; then test -x "$path" || exit 1; fi
    done
  ' sh {} +
sudo -u _m11p8 -g _m11p8 /bin/test -r "$release/.git/index"
worker_status=$(sudo -u _m11p8 -g _m11p8 /usr/bin/env -i \
  PATH=/usr/bin:/bin LC_ALL=C GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_OPTIONAL_LOCKS=0 /usr/bin/git -C "$release" -c "safe.directory=$release" \
  -c core.fsmonitor=false -c core.untrackedCache=false status --porcelain=v1 --untracked-files=all)
test -z "$worker_status"

as_p8() {
  (cd / && sudo /usr/bin/env -i HOME="$worker_home" PATH=/usr/bin:/bin LC_ALL=C TMPDIR=/private/tmp \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$python" -I -B "$@")
}
approval="$stage/final-a20.json"
as_p8 - "$release" "$rc" "$ci" > "$approval" <<'PY'
import json
import runpy
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
from decision_intelligence.customer_value import p8_candidate, p8_private
from decision_intelligence.customer_value import predictive_protocol as p

runner = runpy.run_path(str(repo / "scripts/generate_m11_evidence.py"))
p.require_successor_amendment()
assert not p.validate_protocol()
owner = p8_private._ROOT.lstat().st_uid
assert owner != 0
private = p8_private.open_private_evidence(repo, owner=owner)
attempt = p8_candidate.A20_ATTEMPT
runner["_require_private_available"](private, private.run_dir(attempt))
history = runner["verify_successor_history"](private)
tip = p8_candidate.verify_a19_consumption(private, require_unused=True)
record = p8_candidate.release_binding(repo, attempt=attempt, manifest=p.protocol_digest(),
                                      code=p.protocol_code_digest(), worker_uid=owner)
assert record["commit"] == sys.argv[2] and record["history_sha256"] == history
assert record["a19_journal_tip_sha256"] == tip
assert sys.argv[3].isdigit() and int(sys.argv[3]) > 0
record.update(ci_run_id=sys.argv[3], host_preflight_passed=True)
print(json.dumps(record, indent=2, sort_keys=True))
PY
```

The unsigned record and A19 journal tip stay private. The gate verifies the A19 claim and terminal row read-only and requires the prior history pin and A19 protocol digests. A missing, mismatched or already extended journal stops preflight. Never initialise or rewrite the journal.

## 3. Separate signature, signed preflight and one dispatch

The independent signer reviews the binding and immediate host preflight before authorising a single launch. Set `OWNER_SIGNING_KEY` to the existing trusted key on the signing host, and keep the signed materials private. Install only after approval.

```zsh
/usr/bin/ssh-keygen -Y sign -f "$OWNER_SIGNING_KEY" -n m11-p8-final-a20 "$approval"
/usr/bin/ssh-keygen -Y verify -f "$trust/allowed_signers" -I m11-p8-owner \
  -n m11-p8-final-a20 -s "$approval.sig" < "$approval"
for file in final-a20.json final-a20.json.sig; do
  sudo test ! -e "$trust/$file"
  sudo test ! -L "$trust/$file"
done
sudo install -o root -g wheel -m 0444 "$approval.sig" "$trust/final-a20.json.sig"
sudo install -o root -g wheel -m 0444 "$approval" "$trust/final-a20.json"

as_p8 "$release/scripts/generate_m11_evidence.py" --preflight 2> "$stage/preflight.stderr"
```

Require the signed preflight PASS. It uses the native supervisor, the configured worker's effective UID and GID, and the installed signature. It checks the index, private staging, and historical chain without creating a claim, reservation or data. On failure, review the stage-specific diagnostic in `preflight.stderr` and stop. Once preflight passes and host conditions are still current, use the separate signed authority to dispatch once.

```zsh
manifest=$("$python" -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$approval")
code=$("$python" -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_sha256"])' "$approval")
(cd / && caffeinate -is sudo /usr/bin/env -i HOME="$worker_home" PATH=/usr/bin:/bin LC_ALL=C \
  TMPDIR=/private/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$python" -I -B \
  "$release/scripts/generate_m11_evidence.py" --mode final \
  --expect-manifest "$manifest" --expect-code "$code") 2>&1 | tee "$stage/supervisor.log"
```

The runner repeats the signed worker check before committing consumption. If consumption occurs, the identity is permanently used. Do not retry after failure or interruption. Observe supervisor status without opening outcome files. After a successful exit and separate owner authorisation for evidence access, verify the private reports and raw file hashes:

```zsh
as_p8 "$release/scripts/generate_m11_evidence.py" --verify \
  "$worker_home/Library/Application Support/M11-P8-Private/staging/m11-p8-final-a20"
```
