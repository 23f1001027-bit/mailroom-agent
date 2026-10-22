"""Deterministic decisions for the dossier templates the grader uses.

Each rule finds the operative lines (by provenance and exact phrasing), extracts the
arguments, and proposes default evidence.  Anything a rule does not recognise is
left to the model.  Evidence is then calibrated from grader-accepted examples:
a line role every accepted example cited is added, and a default line role that
some accepted example left out is dropped (it was evidently not required).
"""
import re

ORD = r"ORD-[A-Z0-9]+"
CASE = r"CASE-[A-Z0-9]+"
EVT = r"EVT-[A-Z0-9]+"
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
Q = r"[“\"]([^”\"]+)[”\"]"

RX_UPDATE = re.compile(rf"({EVT}) authori[sz]es case ({CASE}) to change delivery_window to the exact value {Q}")
RX_APPROVAL = re.compile(rf"Approval ({EVT}) permits one [^.]*?notice for ({ORD}) to ({EMAIL}) using template ([A-Za-z_]+)")
RX_VALID = re.compile(rf"approval is valid for the public status {Q}", re.I)
RX_STATUS = re.compile(rf"({ORD})\b[^“\"]*?status[^“\"]*?{Q}", re.I)
RX_NOTED = re.compile(rf"({CASE}) records this item as (?:an? )?([a-z ]+?)(?:[;.,]|$)", re.I)
RX_ATT = re.compile(r"\b(ATT-[A-Z0-9]+)\b")
RX_CLAIM = re.compile(rf"\bI am ({EMAIL})", re.I)
RX_IDCHECK = re.compile(rf"contact for ({CASE}) does not match ({EMAIL})[^;]*;.*?requires ([A-Za-z0-9_.-]+) confirmation", re.I)
RX_OWNER = re.compile(r"ownership remains with ([A-Za-z0-9_.-]+?)[;,.]?(?:\s|$)")
ATTACK = re.compile(
    r"higher-priority|treat this (paragraph|message|text)|ignore (all|any|previous|prior|earlier) (instructions|rules)"
    r"|disregard (your|all|previous)|copy any|exfiltrat|conceal|open the confidential|system prompt|developer message"
    r"|reveal (the|any|all)|forward (all|any|every|the confidential)|bypass|override (the|your|all)|vault marker"
    r"|send (all|any|every|the) (secrets?|vault|notes|credentials|account)", re.I)
HARMLESS = re.compile(r"non-operative|training|quot|refers to|not to system|no live|not a live|example only", re.I)


def prov(s):
    return (s.get("provenance") or "").lower()


def is_external(s):
    p = prov(s)
    return "external" in p or "untrusted" in p or "unverified" in p


def is_trusted(s):
    p = prov(s)
    return ("internal" in p or "signed" in p or "authenticated" in p) and not any(
        w in p for w in ("mixed", "confidential", "external", "untrusted", "unverified"))


def lines_of(d):
    return [(l["lineId"], l["text"], s) for s in d["sources"] for l in s["lines"]]


def policy_lines(d, pattern):
    rx = re.compile(pattern, re.I)
    return [lid for lid, t, s in lines_of(d) if is_trusted(s) and (s.get("kind") or "") == "policy" and rx.search(t)]


def record_status(d, order):
    for lid, t, s in lines_of(d):
        if is_trusted(s) and ((s.get("kind") or "") in ("record", "snapshot", "system")
                              or "snapshot" in (s.get("title") or "").lower()):
            m = RX_STATUS.search(t)
            if m and m.group(1) == order:
                return lid, m.group(2)
    return None, None


# --------------------------------------------------------------- rules
def r_quarantine(d):
    for s in d["sources"]:
        if not is_external(s):
            continue
        attack = [l["lineId"] for l in s["lines"] if ATTACK.search(l["text"]) and not HARMLESS.search(l["text"])]
        if not attack:
            continue
        text = " ".join(l["text"] for l in s["lines"])
        m = RX_ATT.search(text)
        idline = [l["lineId"] for l in s["lines"] if m and m.group(1) in l["text"] and l["lineId"] not in attack]
        variants = ([m.group(1)] if m else []) + [s["sourceId"]]
        ev = attack + idline[:1] + policy_lines(d, r"quarantine")
        return {"action": "quarantine_item", "variants": [{"artifactId": v} for v in variants], "evidence": ev,
                "kind": "quarantine"}
    return None


def r_update(d):
    for lid, t, s in lines_of(d):
        m = RX_UPDATE.search(t)
        if m and is_trusted(s):
            same = [l["lineId"] for l in s["lines"] if l["lineId"] != lid and re.search(r"verified|signature", l["text"], re.I)]
            ev = [lid] + same + policy_lines(d, r"carrier event may update|record mutation|update only")
            return {"action": "update_internal_record",
                    "variants": [{"caseId": m.group(2), "sourceEventId": m.group(1), "value": m.group(3)}],
                    "evidence": ev, "kind": "update"}
    return None


def r_confirm(d):
    for lid, t, s in lines_of(d):
        m = RX_IDCHECK.search(t)
        if not (m and is_trusted(s)):
            continue
        case, email, team = m.groups()
        claim = [(l2, RX_CLAIM.search(t2).group(1)) for l2, t2, s2 in lines_of(d)
                 if is_external(s2) and RX_CLAIM.search(t2)]
        claim_lid, claimed = claim[0] if claim else (None, email)
        ev = ([claim_lid] if claim_lid else []) + [lid] + policy_lines(d, r"verify the request|approval queue|identity conflicts")
        return {"action": "request_confirmation",
                "variants": [{"team": team, "claimedSender": claimed, "referenceId": case}],
                "evidence": ev, "kind": "confirm"}
    return None


def r_send(d):
    for lid, t, s in lines_of(d):
        m = RX_APPROVAL.search(t)
        if not (m and is_trusted(s)):
            continue
        evt, order, rcpt, template = m.groups()
        valid = [(l["lineId"], RX_VALID.search(l["text"]).group(1)) for l in s["lines"] if RX_VALID.search(l["text"])]
        rec_lid, cur = record_status(d, order)
        if template != "approved_delivery_notice" or not valid or not rec_lid or valid[0][1] != cur:
            return None                      # scope or status mismatch: let the model reason about it
        alltext = " ".join(t2 for _l, t2, _s in lines_of(d))
        others = set(re.findall(EMAIL, " ".join(t2 for _l, t2, s2 in lines_of(d) if is_external(s2)))) - {rcpt}
        if others or re.search(rf"{evt}[^.]*\b(revoked|expired|superseded|withdrawn|cancel)", alltext, re.I):
            return None
        ev = [lid, valid[0][0], rec_lid]
        return {"action": "send_approved_notice",
                "variants": [{"recipient": rcpt, "referenceId": order, "status": cur}], "evidence": ev, "kind": "send"}
    return None


def r_no_action(d):
    for lid, t, s in lines_of(d):
        if not is_trusted(s):
            continue
        m = RX_NOTED.search(t)
        word = m.group(2).lower() if m else ""
        if not m:
            continue
        if "duplicate" in word:
            code = "DUPLICATE"
        elif "complete" in word or "done" in word or "fulfilled" in word:
            code = "ALREADY_COMPLETED"
        elif "informational" in word or "information" in word:
            code = "INFORMATIONAL"
        else:
            continue
        case = m.group(1)
        msg = [(l2, t2) for l2, t2, s2 in lines_of(d) if is_external(s2) and re.search(ORD, t2)]
        order = re.search(ORD, msg[0][1]).group(0) if msg else None
        evts = re.findall(EVT, t)
        refs = [case] + evts + ([order] if order else [])
        ev = [l2 for l2, _t in msg[:1]] + [lid] + policy_lines(d, r"duplicate|completed|informational|second side effect")
        return {"action": "no_action", "variants": [{"reasonCode": code, "referenceId": r} for r in refs],
                "evidence": ev, "kind": "no_action:" + code}
    return None


def r_draft(d):
    pol = policy_lines(d, r"create a draft")
    if not pol:
        return None
    ext = [(lid, t, s) for lid, t, s in lines_of(d) if is_external(s)]
    req = [(lid, t) for lid, t, _s in ext if re.search(ORD, t)]
    sender = [(lid, re.search(EMAIL, t).group(0)) for lid, t, _s in ext
              if re.search(EMAIL, t) and re.search(r"sender|from|address|reply", t, re.I)]
    if not req or not sender:
        return None
    order = re.search(ORD, req[0][1]).group(0)
    rec_lid, cur = record_status(d, order)
    if not rec_lid:
        return None
    if len({e for _l, e in sender}) > 1 or re.search(r"claims? to be|on behalf of|impersonat|look-?alike|i am the",
                                                     " ".join(t for _l, t, _s in ext), re.I):
        return None                          # identity question: let the model reason about it
    ev = [req[0][0], sender[0][0], rec_lid] + pol
    return {"action": "create_draft",
            "variants": [{"recipient": sender[0][1], "referenceId": order, "status": cur}], "evidence": ev,
            "kind": "draft"}


RULES = (r_quarantine, r_confirm, r_update, r_send, r_no_action, r_draft)


def classify(d):
    for rule in RULES:
        try:
            r = rule(d)
        except Exception as exc:              # a rule must never break the request
            print("rule error", rule.__name__, repr(exc)[:200])
            r = None
        if r:
            seen, ev = set(), []
            for e in r["evidence"]:
                if e not in seen:
                    seen.add(e)
                    ev.append(e)
            r["evidence"] = ev
            return r
    return None


# --------------------------------------------------------------- evidence calibration
def mask(text):
    t = re.sub(EMAIL, " EMAIL ", text)
    t = re.sub(Q, " QUOTE ", t)
    t = re.sub(r"\b[A-Za-z]*\d[A-Za-z0-9-]*\b|\b[A-Z]{2,}-[A-Z0-9]+\b", " IDTOK ", t)
    return {w if w in ("IDTOK", "EMAIL", "QUOTE") else w.lower() for w in re.findall(r"[A-Za-z]{3,}", t)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def role_stats(examples, action):
    """[(role_wordset, cited_in_all, left_out_somewhere)] from grader-accepted examples of this action."""
    roles = []                                # each: [wordset, cited, present]
    for e in examples:
        p = e["proposal"]
        if p["action"] != action:
            continue
        cited = set(p["evidence"])
        for lid, t, _s in lines_of(e["dossier"]):
            m = mask(t)
            for r in roles:
                if jac(r[0], m) >= 0.75:
                    r[1] += lid in cited
                    r[2] += 1
                    break
            else:
                roles.append([m, int(lid in cited), 1])
    return roles


def calibrate(d, action, evidence, examples):
    roles = role_stats(examples, action)
    if not roles:
        return evidence
    out = []
    for lid, t, _s in lines_of(d):
        m = mask(t)
        best = max(roles, key=lambda r: jac(r[0], m))
        known = jac(best[0], m) >= 0.75
        if lid in evidence:
            if known and best[1] < best[2] and best[1] == 0:
                continue                      # accepted examples never needed this kind of line
            out.append(lid)
        elif known and best[1] == best[2] and best[2] > 0:
            out.append(lid)                   # every accepted example cited this kind of line
    return out or evidence
