"""Spike: is the finite abstraction small enough, and is the shield non-trivial and sound?

Answers ADR 0001 section "Finite abstraction (F0)". Pure standard library, no sibling imports.
SYNTHETIC toy model of the confined-space crusher job and a two-pump availability game.
NOT the production implementation: in the build, the transition relation used by the model checker
must be written independently of the one used for synthesis (packet P5b). Here both share `step`.

Run: python shield_feasibility.py
Prints state counts, winning-region sizes, shield permissiveness, deadlock counts per perception
resolution assumption, the independent reachability check, and a sabotage check.
"""

from __future__ import annotations

import time
from collections import deque
from itertools import product

V = "VIOL"  # sink state: a covered property was violated

# ----------------------------------------------------------------------------- crusher component
OPS = ("RUN", "STOP", "LOCKED", "ISO", "WORK", "FAIL")
PERM = ("NONE", "VALID", "REVAL")  # REVAL = needs revalidation after a handover
ACTS = ("NOOP", "START", "STOP", "LOCK", "PROVE", "ISSUE", "REVAL", "BEGIN", "TEST",
        "ENTER", "EXIT", "COMPLETE", "CLOSE", "UNISO")
FORCED = {"STOP", "EXIT"}  # assumption A_exec: these are always executed; others may be refused
HMAX = 3  # health bins 0..3; 3 = critical


def c_states():
    for op, h, p, occ, gas, alarm, soon in product(OPS, range(HMAX + 1), PERM, (0, 1),
                                                    (1, 2, 3), (0, 1), (0, 1)):
        yield (op, h, p, occ, gas, alarm, soon)


def c_step(s, a):
    """All environment outcomes of controller action `a` in state `s`, as a set of states / V."""
    op, h, p, occ, gas, alarm, soon = s
    out = set()
    for refused in ((False, True) if (a not in FORCED and a != "NOOP") else (False,)):
        e = "NOOP" if refused else a
        o, p1, occ1, h1 = op, p, occ, h
        tested, viol = False, False
        if e == "START":
            if o == "STOP":
                o = "RUN"
            elif o in ("LOCKED", "ISO", "WORK"):
                viol = True  # energising under lock
        elif e == "STOP" and o == "RUN":
            o = "STOP"
        elif e == "LOCK" and o in ("STOP", "FAIL"):
            o = "LOCKED"
        elif e == "PROVE" and o == "LOCKED":
            o = "ISO"
        elif e == "ISSUE" and p1 == "NONE" and o == "ISO":
            p1 = "VALID"
        elif e == "BEGIN":
            if o == "ISO" and p1 == "VALID":
                o = "WORK"
            elif o in ("STOP", "LOCKED", "RUN", "FAIL") or (o == "ISO" and p1 != "VALID"):
                viol = True  # work on an energised or unpermitted asset
        elif e == "TEST" and o == "WORK":
            tested = True
        elif e == "ENTER" and occ1 == 0:
            if o == "WORK" and p1 == "VALID" and gas == 1 and alarm == 0:
                occ1 = 1
            else:
                viol = True
        elif e == "EXIT":
            occ1 = 0
        elif e == "COMPLETE":
            if occ1 == 1:
                viol = True
            elif o == "WORK":
                o, h1 = "ISO", 0
        elif e == "CLOSE":
            if o == "WORK":
                viol = True
            elif o == "ISO" and p1 != "NONE":
                p1 = "NONE"
        elif e == "UNISO" and o == "ISO":
            if p1 != "NONE":
                viol = True
            else:
                o = "STOP"
        gas1 = 3 if e == "EXIT" else gas
        if viol:
            out.add(V)
            continue
        holders = (0, 1) if e == "REVAL" and p1 == "REVAL" else (0,)
        for holder in holders:
            p2 = "VALID" if (e == "REVAL" and p1 == "REVAL" and holder) else p1
            for bad in ((0, 1) if tested else (0,)):
                for anew in ((0, 1) if (occ1 == 1 and not tested) else (0,)):
                    alarm2 = bad if tested else (1 if (alarm or anew) else 0)
                    healths = [(h1, o, False)]
                    if o == "RUN":
                        if h1 == HMAX:
                            healths = [(h1, o, False), (h1, "FAIL", True)]
                        else:
                            healths = [(h1, o, False), (h1 + 1, o, False)]
                    for h2, o2, failed in healths:
                        if failed:
                            out.add(V)  # uncontrolled failure of the critical asset while running
                            continue
                        for soon_new in ((0, 1) if soon == 0 else (0,)):
                            p3, soon2 = p2, soon_new
                            if soon == 1:
                                soon2 = 0
                                if p3 != "NONE":
                                    p3 = "REVAL"
                            g2 = 1 if tested else min(gas1 + 1, 3)
                            if occ1 == 1 and (g2 == 3 or p3 == "REVAL" or alarm2 == 1):
                                out.add(V)
                            else:
                                out.add((o2, h2, p3, occ1, g2, alarm2, soon2))
    return out


# ----------------------------------------------------------------------------- generic solver
def solve(states, acts, step):
    """Greatest fixed point of the safety game; returns W and safe-action table A_W."""
    W = set(states)
    changed = True
    while changed:
        changed = False
        for s in list(W):
            if not any(all(t != V and t in W for t in step(s, a)) for a in acts):
                W.discard(s)
                changed = True
    safe = {s: frozenset(a for a in acts if all(t != V and t in W for t in step(s, a))) for s in W}
    return W, safe


def intervals(n):
    return [(lo, hi) for lo in range(n) for hi in range(lo, n)]


def obs(s):  # everything except the hidden health bin
    return (s[0],) + s[2:]


def build_shield(W, safe, max_width):
    """A(o, B) = intersection over h in B (with (o,h) in W) of safe(o,h)."""
    table, dead, total = {}, [], 0
    by_o = {}
    for s in W:
        by_o.setdefault(obs(s), {})[s[1]] = safe[s]
    for o, hs in by_o.items():
        for lo, hi in intervals(HMAX + 1):
            if hi - lo + 1 > max_width:
                continue
            members = [hs[h] for h in range(lo, hi + 1) if h in hs]
            if not members:
                continue
            total += 1
            acts = frozenset.intersection(*members)
            table[(o, (lo, hi))] = acts
            if not acts:
                dead.append((o, (lo, hi)))
    return table, dead, total


def check(initial, table, step_fn, max_width, drop=None):
    """Independent reachability on the shielded product (o,h,B). Contract C1: h in B, |B|<=width."""
    seen, queue, bad = set(), deque(), []
    for B in intervals(HMAX + 1):
        if B[1] - B[0] + 1 <= max_width and B[0] <= initial[1] <= B[1]:
            queue.append((initial, B))
    while queue:
        s, B = queue.popleft()
        if (s, B) in seen:
            continue
        seen.add((s, B))
        allowed = table.get((obs(s), B), frozenset())
        if drop and (obs(s), B) == drop[0]:
            allowed = allowed | {drop[1]}  # sabotage: add an unsafe action to the shield
        if not allowed:
            bad.append(("DEADLOCK", s, B))
            continue
        for a in allowed:
            for t in step_fn(s, a):
                if t == V:
                    bad.append(("VIOLATION", s, a))
                    continue
                for B2 in intervals(HMAX + 1):
                    if B2[1] - B2[0] + 1 <= max_width and B2[0] <= t[1] <= B2[1]:
                        queue.append((t, B2))
    return seen, bad


# ----------------------------------------------------------------------------- pump pair
PO = ("RUN", "STOP", "ISO", "WORK", "FAIL")
PA = ("NOOP", "START", "STOP", "ISOLATE", "BEGIN", "COMPLETE", "UNISO")


def pump_step1(op, h, a):
    """Per-pump outcomes: set of (op, h) or V."""
    out = set()
    for refused in ((False, True) if a not in FORCED and a != "NOOP" else (False,)):
        e = "NOOP" if refused else a
        o, h1 = op, h
        if e == "START" and o == "STOP":
            o = "RUN"
        elif e == "START" and o in ("ISO", "WORK"):
            out.add(V)
            continue
        elif e == "STOP" and o == "RUN":
            o = "STOP"
        elif e == "ISOLATE" and o in ("STOP", "FAIL"):
            o = "ISO"
        elif e == "BEGIN":
            if o == "ISO":
                o = "WORK"
            elif o in ("RUN", "STOP", "FAIL"):
                out.add(V)
                continue
        elif e == "COMPLETE" and o == "WORK":
            o, h1 = "ISO", 0
        elif e == "UNISO" and o == "ISO":
            o = "STOP"
        if o == "RUN":
            out.add((o, h1))
            out.add((o, h1 + 1) if h1 < HMAX else ("FAIL", h1))
        else:
            out.add((o, h1))
    return out


def pair_step(s, a):
    (o1, h1, o2, h2), (a1, a2) = s, a
    res = set()
    for t1, t2 in product(pump_step1(o1, h1, a1), pump_step1(o2, h2, a2)):
        if V in (t1, t2) or (t1[0] in ("ISO", "WORK", "FAIL") and t2[0] in ("ISO", "WORK", "FAIL")):
            res.add(V)
        else:
            res.add(t1 + t2)
    return res


def main():
    t0 = time.perf_counter()
    cs = list(c_states())
    W, safe = solve(cs, ACTS, c_step)
    print(f"crusher raw states {len(cs)}, winning region {len(W)}, "
          f"solve {time.perf_counter() - t0:.2f} s")
    n_o = len({obs(s) for s in W})
    print(f"observable abstract states in W: {n_o}")
    avg = sum(len(safe[s]) for s in W) / len(W)
    print(f"mean number of safe actions per winning state: {avg:.2f} of {len(ACTS)}")
    init = ("RUN", 0, "NONE", 0, 3, 0, 0)
    print("initial state in W:", init in W)

    for width in (1, 2, 3, 4):
        table, dead, total = build_shield(W, safe, width)
        seen, bad = check(init, table, c_step, width)
        kinds = {}
        for k, *_ in bad:
            kinds[k] = kinds.get(k, 0) + 1
        only_wide = [len(v) for (o, B), v in table.items() if B[1] - B[0] + 1 == width]
        print(f"max |B|={width}: shield entries {total}, deadlocks {len(dead)}, "
              f"reachable product states {len(seen)}, checker findings {kinds or 'none'}; "
              f"mean safe actions at |B|={width}: {sum(only_wide) / len(only_wide):.2f}")
        if width == 2:
            table2, seen2 = table, seen
    # sabotage: add the unsafe BEGIN action to a reachable shield entry for a LOCKED asset
    victim = next((obs(s), B) for (s, B) in sorted(seen2, key=str)
                  if s[0] == "LOCKED" and "BEGIN" not in table2[(obs(s), B)])
    _, bad = check(init, table2, c_step, 2, drop=(victim, "BEGIN"))
    print("sabotage (BEGIN admitted on a LOCKED, unproven asset) detected:",
          bool(bad) and any(k == "VIOLATION" for k, *_ in bad))

    ps = [(a, b, c, d) for a, b, c, d in product(PO, range(HMAX + 1), PO, range(HMAX + 1))]
    pacts = list(product(PA, PA))
    t1 = time.perf_counter()
    PW, _ = solve(ps, pacts, pair_step)
    print(f"pump pair raw states {len(ps)}, winning region {len(PW)}, "
          f"joint actions {len(pacts)}, solve {time.perf_counter() - t1:.2f} s")


if __name__ == "__main__":
    main()
