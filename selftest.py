"""Protocol self-test. Run the server with MOCK_MODEL=1 first, then:
    python selftest.py http://127.0.0.1:8030/
It plays the grader: propose, signed receipts, commit, replay, conflicts, bad input.
It never calls the real model and never uses one of your 10 grader checks."""
import base64
import hashlib
import json
import sys
import uuid

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8030/"
PROFILE = "ga5-mailroom-action-gate/v2"


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def dossier(i, text=None):
    return {"dossierId": f"D{i}", "partition": "stable_core", "receivedAt": "2026-05-01T10:00:00Z",
            "mailbox": "support@acme.test", "objective": "triage this mail",
            "sources": [{"sourceId": f"S{i}", "kind": "email", "provenance": "external", "title": "Hi",
                         "lines": [{"lineId": f"L{i}a", "text": text or f"ORD-{i} where is my parcel"},
                                   {"lineId": f"L{i}b", "text": "thanks"}]}]}


priv = Ed25519PrivateKey.generate()
x = base64.urlsafe_b64encode(priv.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode().rstrip("=")
JWK = {"kty": "OKP", "crv": "Ed25519", "x": x}
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra if not cond else "")


def post(body, raw=None):
    if raw is not None:
        return httpx.post(URL, content=raw, headers={"Content-Type": "application/json"}, timeout=60)
    return httpx.post(URL, json=body, timeout=60)


def propose_body(eid, ds):
    return {"profile": PROFILE, "operation": "propose", "evaluationId": eid,
            "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": JWK},
            "corpus": {"coreId": "c", "auditId": "a", "stableCount": len(ds), "freshCount": 0},
            "allowedActions": ["create_draft", "update_internal_record", "send_approved_notice",
                               "request_confirmation", "quarantine_item", "no_action"],
            "dossiers": ds}


def digest_of(p):
    view = {k: p[k] for k in ("dossierId", "callId", "action", "target", "payload")}
    view["evidence"] = sorted(p["evidence"])
    return sha(canon(view))


def receipt(eid, digest, p, accepted=True):
    inner = {"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"], "accepted": accepted,
             "proposalDigest": digest_of(p), "receiptId": uuid.uuid4().hex}
    msg = canon({"profile": PROFILE, "evaluationId": eid, "inputDigest": digest, "receipt": inner}).encode()
    return {**inner, "receiptSignature": base64.b64encode(priv.sign(msg)).decode()}


def commit_body(eid, digest, receipts):
    return {"profile": PROFILE, "operation": "commit", "evaluationId": eid, "inputDigest": digest,
            "receipts": receipts}


eid = "eval-" + uuid.uuid4().hex[:8]
ds = [dossier(i) for i in range(1, 6)]
r = post(propose_body(eid, ds))
check("propose 200", r.status_code == 200, r.text[:200])
res = r.json()
check("inputDigest matches", res["inputDigest"] == sha(canon(ds)))
check("one proposal per dossier", [p["dossierId"] for p in res["proposals"]] == [d["dossierId"] for d in ds])
check("callIds unique and valid", len({p["callId"] for p in res["proposals"]}) == 5
      and all(12 <= len(p["callId"]) <= 128 for p in res["proposals"]))
r2 = post(propose_body(eid, ds))
check("propose replay is identical", r2.status_code == 200 and r2.content == r.content)

ds2 = [dossier(i) for i in range(1, 6)]
ds2[0] = dossier(1, "CHANGED text")
check("changed content -> 409", post(propose_body(eid, ds2)).status_code == 409)

eid2 = "eval-" + uuid.uuid4().hex[:8]
res2 = post(propose_body(eid2, ds)).json()
check("stable callIds across evaluations",
      [p["callId"] for p in res2["proposals"]] == [p["callId"] for p in res["proposals"]]
      and [p["payload"] for p in res2["proposals"]] == [p["payload"] for p in res["proposals"]])

props, dg = res["proposals"], res["inputDigest"]
good = [receipt(eid, dg, p, accepted=(i % 2 == 0)) for i, p in enumerate(props)]

bad = json.loads(json.dumps(good))
bad[0]["receiptSignature"] = good[1]["receiptSignature"]
check("moved signature rejected", post(commit_body(eid, dg, bad)).status_code == 400)
bad = json.loads(json.dumps(good))
bad[2]["accepted"] = not bad[2]["accepted"]
check("flipped accepted rejected", post(commit_body(eid, dg, bad)).status_code == 400)
check("missing receipt rejected", post(commit_body(eid, dg, good[:-1])).status_code == 400)
check("duplicate receipt rejected", post(commit_body(eid, dg, good[:-1] + [good[0]])).status_code == 400)
bad = json.loads(json.dumps(good))
bad[1]["proposalDigest"] = "0" * 64
check("wrong proposalDigest rejected", post(commit_body(eid, dg, bad)).status_code == 400)
bad = json.loads(json.dumps(good))
bad[0]["extra"] = "x"
check("unknown receipt field rejected", post(commit_body(eid, dg, bad)).status_code == 400)
bad = json.loads(json.dumps(good))
bad[0]["receiptSignature"] = "!!!notbase64!!!"
check("garbage signature rejected", post(commit_body(eid, dg, bad)).status_code == 400)
bad = json.loads(json.dumps(good))
sg = bad[0]["receiptSignature"]
alt = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
last = sg.rstrip("=")[-1]
bad[0]["receiptSignature"] = sg.rstrip("=")[:-1] + alt[alt.index(last) ^ 1] + sg[len(sg.rstrip("=")):]
check("padding-bit forged signature rejected", post(commit_body(eid, dg, bad)).status_code == 400)
check("unknown evaluation rejected", post(commit_body("nope-" + eid, dg, good)).status_code == 400)
check("wrong inputDigest rejected", post(commit_body(eid, "0" * 64, good)).status_code == 400)

c = post(commit_body(eid, dg, good))
check("commit 200 completed", c.status_code == 200 and c.json()["status"] == "completed", c.text[:200])
st = [o["status"] for o in c.json()["outcomes"]]
check("executed only when accepted", st == ["executed" if i % 2 == 0 else "rejected" for i in range(5)], st)
c2 = post(commit_body(eid, dg, good))
check("commit replay identical", c2.status_code == 200 and c2.content == c.content)
other = [receipt(eid, dg, p, accepted=True) for p in props]
check("different receipts after commit -> 409", post(commit_body(eid, dg, other)).status_code == 409)

eid3 = "eval-" + uuid.uuid4().hex[:8]
res3 = post(propose_body(eid3, ds)).json()
reuse = [receipt(eid3, res3["inputDigest"], p) for p in res3["proposals"]]
reuse[0]["receiptId"] = good[0]["receiptId"]
inner = {k: v for k, v in reuse[0].items() if k != "receiptSignature"}
reuse[0]["receiptSignature"] = base64.b64encode(priv.sign(canon({"profile": PROFILE, "evaluationId": eid3,
    "inputDigest": res3["inputDigest"], "receipt": inner}).encode())).decode()
check("nonce from another evaluation rejected", post(commit_body(eid3, res3["inputDigest"], reuse)).status_code == 400)
check("garbage body -> 400", post(None, raw="not json").status_code == 400)
check("array body -> 400", post(None, raw="[1,2]").status_code == 400)
check("bad operation -> 400", post({"profile": PROFILE, "operation": "delete", "evaluationId": "x"}).status_code == 400)
check("duplicate dossierId -> 400",
      post(propose_body("eval-" + uuid.uuid4().hex[:8], [dossier(1), dossier(1)])).status_code == 400)
check("empty dossiers -> 400", post(propose_body("eval-" + uuid.uuid4().hex[:8], [])).status_code == 400)
nokey = propose_body("eval-" + uuid.uuid4().hex[:8], ds)
del nokey["receiptVerifier"]
check("missing verifier -> 400", post(nokey).status_code == 400)
check("json content-type", r.headers["content-type"].startswith("application/json"))
print("\nALL PASSED" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
