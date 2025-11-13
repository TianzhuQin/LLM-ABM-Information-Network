from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


@dataclass
class Persona:
    pid: int
    gender: str
    race: str
    age: int
    religion: str
    political: str
    extra: Optional[Dict[str, str]] = None


def persona_to_text(p: Persona) -> str:
    base = (
        f"gender={p.gender}, age={p.age}, race/ethnicity={p.race}, "
        f"religion={p.religion}, political affiliation={p.political}"
    )
    if getattr(p, "extra", None):
        more = ", ".join([f"{k}={v}" for k, v in p.extra.items()])
        return base + ", " + more
    return base


def _sample_from_spec(spec: Any) -> Any:
    if isinstance(spec, (str, int, float)):
        return spec
    if isinstance(spec, dict):
        if "choices" in spec:
            choices = list(spec["choices"].keys())
            probs = np.array(list(spec["choices"].values()), dtype=float)
            probs = probs / probs.sum()
            return np.random.choice(choices, p=probs)
        if spec.get("dist") == "normal":
            mean = float(spec.get("mean", 35))
            std = float(spec.get("std", 15))
            mn = float(spec.get("min", 15))
            mx = float(spec.get("max", 80))
            return int(np.clip(np.random.normal(mean, std), mn, mx))
        if "uniform" in spec:
            lo, hi = spec["uniform"]
            return int(np.random.randint(int(lo), int(hi) + 1))
    return spec


def sample_personas(n: int, segments: Optional[List[Dict]] = None) -> List[Persona]:
    people: List[Persona] = []
    if not segments:
        for pid in range(n):
            age = int(np.clip(np.random.normal(35, 15), 15, 80))
            g_r = np.random.random()
            gender = "Man" if g_r < 0.49 else ("Woman" if g_r < 0.98 else "Nonbinary")
            r = np.random.random()
            if r < 0.57:
                race = "White"
            elif r < 0.69:
                race = "Black"
            elif r < 0.88:
                race = "Hispanic"
            elif r < 0.94:
                race = "Asian"
            elif r < 0.95:
                race = "American Indian/Alaska Native"
            else:
                race = "Native Hawaiian/Pacific Islander"
            rel = np.random.random()
            if race == "White":
                religion = "Protestant" if rel < 0.49 else ("Catholic" if rel < 0.69 else "Unreligious")
            elif race == "Black":
                religion = "Protestant" if rel < 0.68 else ("Catholic" if rel < 0.75 else "Unreligious")
            elif race == "Hispanic":
                religion = "Catholic" if rel < 0.76 else ("Protestant" if rel < 0.88 else "Unreligious")
            elif race in ["Asian", "Native Hawaiian/Pacific Islander"]:
                if rel < 0.16:
                    religion = "Protestant"
                elif rel < 0.30:
                    religion = "Catholic"
                elif rel < 0.37:
                    religion = "Muslim"
                elif rel < 0.44:
                    religion = "Buddhist"
                elif rel < 0.59:
                    religion = "Hindu"
                else:
                    religion = "Unreligious"
            else:
                religion = "Protestant" if rel < 0.47 else ("Catholic" if rel < 0.58 else "Unreligious")
            pol = np.random.random()
            if race == "White":
                political = "Republican" if pol < 0.56 else "Democrat"
            elif race == "Black":
                political = "Republican" if pol < 0.12 else "Democrat"
            elif race == "Hispanic":
                political = "Republican" if pol < 0.35 else "Democrat"
            elif race in ["Asian", "Native Hawaiian/Pacific Islander"]:
                political = "Republican" if pol < 0.37 else "Democrat"
            else:
                political = "Republican" if pol < 0.4 else "Democrat"
            people.append(Persona(pid, gender, race, age, religion, political, extra=None))
        return people

    props = np.array([max(0.0, float(s.get("proportion", 0))) for s in segments], dtype=float)
    if props.sum() <= 0:
        props = np.ones(len(segments), dtype=float)
    props = props / props.sum()
    counts = (props * n).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        fracs = props * n - counts
        order = np.argsort(-fracs)
        for idx in order[:remainder]:
            counts[idx] += 1

    pid = 0
    for seg_idx, (seg, cnt) in enumerate(zip(segments, counts)):
        traits = seg.get("traits", {}) or {}
        for _ in range(int(cnt)):
            gender = _sample_from_spec(traits.get("gender", {"choices": {"Man": 0.49, "Woman": 0.49, "Nonbinary": 0.02}}))
            race = _sample_from_spec(
                traits.get(
                    "race",
                    {
                        "choices": {
                            "White": 0.57,
                            "Black": 0.12,
                            "Hispanic": 0.19,
                            "Asian": 0.06,
                            "American Indian/Alaska Native": 0.01,
                            "Native Hawaiian/Pacific Islander": 0.01,
                        }
                    },
                )
            )
            age = int(_sample_from_spec(traits.get("age", {"dist": "normal", "mean": 35, "std": 15, "min": 15, "max": 80})))
            default_religion = {
                "choices": {
                    "Protestant": 0.45,
                    "Catholic": 0.35,
                    "Muslim": 0.05,
                    "Buddhist": 0.04,
                    "Hindu": 0.05,
                    "Unreligious": 0.06,
                }
            }
            religion = _sample_from_spec(traits.get("religion", default_religion))
            political = _sample_from_spec(traits.get("political", {"choices": {"Republican": 0.45, "Democrat": 0.55}}))
            extra_keys = [k for k in traits.keys() if k not in ["gender", "race", "age", "religion", "political"]]
            extra = {}
            for k in extra_keys:
                extra[k] = _sample_from_spec(traits[k])
            # annotate segment index and optional custom segment name for downstream grouping
            extra["_segment_index"] = int(seg_idx)
            if "name" in seg and seg["name"] is not None:
                try:
                    extra["_segment_name"] = str(seg["name"])
                except Exception:
                    extra["_segment_name"] = str(seg_idx)
            people.append(Persona(pid, str(gender), str(race), int(age), str(religion), str(political), extra=extra or None))
            pid += 1

    while len(people) < n:
        seg0 = segments[0]
        traits = seg0.get("traits", {}) or {}
        gender = _sample_from_spec(traits.get("gender", "Man"))
        race = _sample_from_spec(traits.get("race", "White"))
        age = int(_sample_from_spec(traits.get("age", {"uniform": [18, 65]})))
        religion = _sample_from_spec(traits.get("religion", "Unreligious"))
        political = _sample_from_spec(traits.get("political", "Democrat"))
        # pad from first segment; mark segment_index=0
        people.append(Persona(pid, str(gender), str(race), int(age), str(religion), str(political), extra={"_segment_index": 0}))
        pid += 1

    return people


def personas_from_graph(G: "Any") -> List[Persona]:
    """Build personas directly from a networkx.Graph's node attributes.
    Expected keys (optional): gender, race, age, religion, political.
    Any other keys are placed into extra.
    """
    people: List[Persona] = []
    try:
        nodes = list(G.nodes())
    except Exception:
        raise TypeError("G must be a networkx.Graph-like object with .nodes()")
    # Try to preserve integer ordering if present
    try:
        nodes = sorted(nodes)
    except Exception:
        pass
    for pid, node in enumerate(nodes):
        try:
            attrs = dict(G.nodes[node])
        except Exception:
            attrs = {}
        gender = str(attrs.pop("gender", "Unknown"))
        race = str(attrs.pop("race", "Unknown"))
        try:
            age = int(attrs.pop("age", 35))
        except Exception:
            age = 35
        religion = str(attrs.pop("religion", "Unreligious"))
        political = str(attrs.pop("political", "Unknown"))
        extra = attrs or None
        people.append(Persona(pid, gender, race, age, religion, political, extra=extra))
    return people



