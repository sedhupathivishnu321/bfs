"""Multilingual rule-based report labeler (EN/ES/DE/NL/HR/TR/BG/FR).

Only 58/4407 training studies carry expert labels; the rest ship a free-text
report in one of eight languages. This module converts reports into *silver*
labels in {0, 0.5, 1} (negative / uncertain / positive) for the 12 targets.

Design (CheXpert-labeler style, extended to multiple languages):
  1. Text normalisation: lower-case, strip Latin diacritics, map Turkish dotless i.
  2. Drop history/indication sentences and questions (they mention the
     *suspected* finding, not the observed one).
  3. Sentence -> sub-clauses (commas, contrastive conjunctions).
  4. A finding term is attributed to the *nearest* structure mention in the
     sentence (both word orders occur: "medial meniscus tear" vs
     "rotura del menisco interno" vs "medijalnog meniska ruptura").
  5. Negation / uncertainty cues are scoped to the sub-clause, plus a
     sentence-initial negation that distributes over lists ("No effusion,
     fracture or contusion").
The rules were written from the vocabulary of *unlabeled* reports; the 58 gold
studies are used only for measurement (see ``evaluate``).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def norm(s: str) -> str:
    s = s.lower().replace("ı", "i").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[ \t]+", " ", s)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("ı", "i"))
    return "".join(c for c in s if not unicodedata.combining(c))


def rx(*alts: str) -> re.Pattern:
    # patterns are written lower-case; only strip accents (lower() would corrupt \W, \S, ...)
    return re.compile("|".join(f"(?:{strip_accents(a)})" for a in alts))


# ----------------------------------------------------------------- cue lexicons
NEG = rx(r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative\b", r"\babsent\b", r"\bfree of\b", r"\bintact\b",
         r"\bnormal", r"\bunremarkable\b", r"\bpreserved\b", r"\bruled out\b", r"\bresolved\b", r"\bnone\b",
         r"\bsin\b", r"\bni\b", r"\bintegr[oa]s?\b", r"\bconservad[oa]s?\b", r"\bausencia\b", r"\bdescart",
         r"\bkein\w*\b", r"\bohne\b", r"\bnicht\b", r"\bintakt\w*\b", r"\bunauffallig\w*\b", r"\bregelrecht\w*\b",
         r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"\bnormaal\b", r"\bgaaf\b",
         r"\bbez\b", r"\bnema\b", r"\bne\b", r"\buredn\w*\b", r"\bocuvan\w*\b", r"\bintaktn\w*\b", r"\bnije\b",
         r"\byok\w*\b", r"\bizlenme\w*\b", r"\bsaptanma\w*\b", r"\bgozlenme\w*\b", r"\bdogal\w*\b",
         r"\bolmaksizin\b", r"\bdegil\w*\b", r"\brastlanma\w*\b", r"\bizlenmemekte\w*\b",
         r"няма", r"\bбез\b", r"\bне\b", r"нормалн", r"запазен", r"интакт", r"б\.о\.", r"липсв",
         r"\bpas de\b", r"\babsence\b", r"\bsans\b", r"\bnon\b")
UNC = rx(r"\bpossibl", r"\bprobabl", r"\bsuspect", r"\bsuspicio", r"\bcannot be excluded\b", r"\blikely\b",
         r"\bmay\b", r"\bquestionable\b", r"\bequivocal\b", r"\bdd\b", r"\bposible\b", r"\bsospech",
         r"\bverdacht", r"\bfraglich", r"\bam ehesten\b", r"\bmogelijk", r"\bverdacht", r"\bwaarschijnlijk",
         r"\bsuspekt", r"\bmoguc", r"\bvjerojatn", r"\bsuphe", r"\bolasi\b", r"\bmuhtemel",
         r"вероятн", r"съмнени", r"възможн", r"\bsuspicion\b", r"\bpodria\b", r"\bkonnte\b",
         r"zou kunnen", r"\bmoze\b", r"olabilir", r"\bможе\b", r"\bpourrait\b", r"\bpodria\b")
CONTRAST = re.compile(r",|\bbut\b|\bhowever\b|\bpero\b|\bsin embargo\b|\baber\b|\bjedoch\b|\bmaar\b|\bechter\b"
                      r"|\bali\b|\bmedutim\b|\bancak\b|\bfakat\b|\bно\b|\bmais\b|\bwhereas\b|\bwhile\b")
HISTORY = rx(r"^\W*(clinical|history|indication|reason|comparison|antecedente|klinische|vraagstel|anamnes"
             r"|indikacij|klinik|клинич|renseignement|motivo|fragestellung|diagnostische)")

# ----------------------------------------------------------------- structures
S = {
    "ACL": rx(r"\bacl\b", r"\blca\b", r"anterior cruciate", r"cruzado anterior", r"\bvkb\b", r"vorder\w* kreuzband",
              r"voorste kruisband", r"prednj\w* (ukriz|kriz)\w*", r"\bon capraz", r"предн\w* кръстн",
              r"croise anterieur", r"\blia\b"),
    "PCL": rx(r"\bpcl\b", r"\blcp\b", r"posterior cruciate", r"cruzado posterior", r"\bhkb\b", r"hinter\w* kreuzband",
              r"achterste kruisband", r"straznj\w* (ukriz|kriz)\w*", r"arka capraz", r"задн\w* кръстн",
              r"croise posterieur", r"\blip\b"),
    "MCL": rx(r"\bmcl\b", r"medial collateral", r"tibial collateral", r"colateral (medial|interno)", r"\blcm\b",
              r"\blli\b", r"innenband", r"medial\w* kollateral", r"mediale collaterale", r"\bmcb\b",
              r"medijaln\w* kolateral", r"медиал\w* колатерал", r"collateral (interne|medial)"),
    "LCL": rx(r"\blcl\b", r"\bfcl\b", r"lateral collateral", r"fibular collateral", r"colateral (lateral|externo)",
              r"\blce\b", r"aussenband", r"lateral\w* kollateral", r"laterale collaterale", r"\blcb\b",
              r"lateraln\w* kolateral", r"латерал\w* колатерал", r"collateral (externe|lateral)", r"\blle\b"),
    "MedMen": rx(r"(medial|internal|interno|interna|mediale[nrs]?|medijaln\w*|медиалн\w*|interne)\W+(\w+\W+)?(menisc|menisk|menisq|менис)",
                 r"(menisc|menisk|menisq|менис)\w*\W+(\w+\W+)?(medial|interno|interna|mediale|medijaln\w*|медиалн\w*|interne)\b",
                 r"innenmenisk"),
    "LatMen": rx(r"(lateral|external|externo|externa|laterale[nrs]?|lateraln\w*|латералн\w*|externe)\W+(\w+\W+)?(menisc|menisk|menisq|менис)",
                 r"(menisc|menisk|menisq|менис)\w*\W+(\w+\W+)?(lateral|externo|externa|laterale|lateraln\w*|латералн\w*|externe)\b",
                 r"aussenmenisk"),
}
# generic meniscus mention with no side -> lets "medial and lateral menisci" etc. resolve via side words
TEAR = rx(r"\btear", r"\btorn\b", r"ruptur", r"disrupt", r"avuls", r"\brotur", r"\brot[oa]\b", r"desgarr",
          r"\briss", r"scheur", r"puknu", r"razdor", r"yirt", r"разкъс", r"руптур", r"dechir", r"flap",
          r"bucket", r"\bdefect", r"transsection", r"\bdiscontinu", r"devamsiz")
TEAR_UNC = rx(r"\blesion", r"\blasion", r"\blaesie", r"\blezij", r"\blezyon", r"увред", r"\binjur", r"ozljed", r"hasar")
SPRAIN = rx(r"sprain", r"strain", r"esguince", r"distensi", r"distorsi", r"zerrung", r"dehnung", r"verstuik",
            r"istegnu", r"burkul", r"gerilme", r"разтяг", r"entorse", r"partial", r"parcial", r"teilruptur",
            r"partie", r"parcijal", r"kismi", r"частичн", r"parziel", r"thicken", r"engros", r"verdick", r"zadeb",
            r"kalinlas", r"удебел", r"edema", r"odem", r"oedeem", r"едем", r"oedeme")

OA = rx(r"osteoart", r"osteophyt", r"osteofit", r"остеофит", r"arthros", r"artros", r"artroz", r"gonart", r"остеоарт", r"артроз", r"\boa\b",
        r"degenerative joint", r"cambios degenerativos articulares")
COMP = {
    "Medial OA": rx(r"\bmedial", r"\binterno\b", r"\binterna\b", r"\binterne\b", r"\bmediale", r"\bmedijaln",
                    r"медиалн", r"\binnen"),
    "Lateral OA": rx(r"\blateral", r"\bexterno\b", r"\bexterna\b", r"\bexterne\b", r"\blaterale", r"\blateraln",
                     r"латерал", r"\baussen"),
    "PF OA": rx(r"patel\w*femor", r"femor\w*patel", r"retropatell", r"пател\w*фемор", r"фемор\w*пател",
                r"\bpf\b", r"patellar", r"\bpatel"),
}
TRI = rx(r"tricompart", r"three compart", r"all (three )?compart", r"3 compart", r"todos los compartimentos",
         r"tricompartimental", r"drei kompart", r"alle (drie )?compartiment", r"sva tri", r"svim odjelj",
         r"tum kompartman", r"uc kompartman", r"трите", r"три компарт", r"trois compart", r"pangonart",
         r"pan-?compart")

SIMPLE = {
    "Effusion": rx(r"effusion", r"derrame", r"erguss", r"hydrops", r"effusie", r"vocht in (het )?gewricht",
                   r"gewrichtsvocht", r"izljev", r"efuzyon", r"(mayi|sivi) artis", r"излив", r"epanchement",
                   r"versamento", r"intraarticular fluid", r"joint fluid"),
    "Synovitis": rx(r"synovit", r"sinovit", r"synovialit", r"синовит", r"synovial (thickening|proliferation|hypertroph)"),
    "Baker's": rx(r"baker", r"popliteal cyst", r"quiste popliteo", r"poplitealzyste", r"popliteacyste",
                  r"poplitealn\w* cist", r"popliteal kist", r"поплит\w* кист", r"kyste poplite",
                  r"gastrocnemi\w*[- ]semimembran\w* burs", r"semimembranos\w*[- ]gastrocnem\w* burs"),
    "Contusion": rx(r"contusi", r"kontusion", r"kontuzij", r"kontuzyon", r"контузи", r"bone bruise", r"kneuzing",
                    r"bone bruising", r"\bbruise"),
    "Fracture": rx(r"fractur", r"fractuur", r"fraktur", r"prijelom", r"\bkirik", r"фрактур", r"счупв", r"breuk",
                   r"\bfx\b"),
}
BME = rx(r"bone marrow (o)?edema", r"marrow (o)?edema", r"edema oseo", r"edema de la medula", r"knochenmark\w*odem",
         r"knochenodem", r"beenmerg\w*oedeem", r"kostan\w* edem", r"kemik iligi odem", r"костно\w* едем",
         r"oedeme (osseux|medullaire|de la moelle)", r"medular\w* edem", r"bone edema")
EFF_TRIVIAL = rx(r"physiolog", r"fisiolog", r"fysiolog", r"fizioloski", r"fizyolojik", r"физиолог", r"trace",
                 r"\bminimal", r"\bminimo\b", r"\bminim\w*", r"\bgering", r"\bminieme", r"\bdiscret")


def sentences(text: str) -> list[str]:
    t = norm(text)
    parts = re.split(r"(?<![0-9])[.;\n]+(?![0-9])|\s{3,}", t)
    return [p.strip() for p in parts if p and p.strip()
            and "?" not in p and not HISTORY.search(p.strip())]


def clauses(sent: str) -> list[tuple[int, int]]:
    bounds, last = [], 0
    for m in CONTRAST.finditer(sent):
        bounds.append((last, m.start()))
        last = m.end()
    bounds.append((last, len(sent)))
    return bounds


def polarity(sent: str, pos: int, lead_neg: bool) -> float:
    """1 = affirmed, 0.5 = uncertain, 0 = negated, for the clause containing `pos`."""
    for a, b in clauses(sent):
        if a <= pos <= b:
            c = sent[a:b]
            if NEG.search(c):
                return 0.0
            if lead_neg and a > 0 and not re.search(r"\w{3,}\s+\w{3,}\s+\w{3,}\s+\w{3,}", c):
                return 0.0  # short list item after a leading "No ..." distributes the negation
            if UNC.search(c):
                return 0.5
            return 1.0
    return 1.0


def nearest(sent: str, pos: int, pats: dict[str, re.Pattern]) -> str | None:
    best, bd = None, 10 ** 9
    for k, p in pats.items():
        for m in p.finditer(sent):
            d = pos - m.end() if m.end() <= pos else (m.start() - pos) * 1.5  # prefer preceding mention
            if d < bd:
                best, bd = k, d
    return best if bd < 250 else None


def label_report(text: str) -> dict[str, float]:
    out = {k: 0.0 for k in LABELS}
    if not isinstance(text, str):
        return out
    struct_map = {"ACL": "ACL", "MCL": "MCL", "MedMen": "Medial Meniscus", "LatMen": "Lateral Meniscus"}
    for s in sentences(text):
        lead_neg = bool(re.match(r"^\W*(no|sin|kein\w*|geen|bez|nema|няма|pas de|absence of|without)\b", s))
        # --- structure injuries
        for pat, ligament_only in ((TEAR, False), (TEAR_UNC, False), (SPRAIN, True)):
            for m in pat.finditer(s):
                st = nearest(s, m.start(), S)
                if st not in struct_map or (ligament_only and st not in ("ACL", "MCL")):
                    continue
                v = polarity(s, m.start(), lead_neg)
                if pat is TEAR_UNC or (ligament_only and pat is SPRAIN and v == 1.0 and st == "ACL"):
                    v = min(v, 0.5)
                out[struct_map[st]] = max(out[struct_map[st]], v)
        # --- osteoarthritis by compartment
        for m in OA.finditer(s):
            v = polarity(s, m.start(), lead_neg)
            if v == 0:
                continue
            if TRI.search(s):
                locs = list(COMP)
            else:
                a, b = next((a, b) for a, b in clauses(s) if a <= m.start() <= b)
                window = s[max(0, a - 80):min(len(s), b + 80)]
                locs = [k for k, p in COMP.items() if p.search(window)] or \
                       ([nearest(s, m.start(), COMP)] if nearest(s, m.start(), COMP) else [])
            for k in locs:
                out[k] = max(out[k], v)
        if TRI.search(s) and re.search(r"degenera|chondro|cartilag|cartilag|knorpel|kraakbeen|hrskav|kikirda|хрущ", s):
            for k in COMP:
                out[k] = max(out[k], 0.5)
        # --- simple findings
        for lab, pat in SIMPLE.items():
            for m in pat.finditer(s):
                v = polarity(s, m.start(), lead_neg)
                if lab == "Effusion" and v > 0 and EFF_TRIVIAL.search(s):
                    v = 0.5 if not re.search(r"physiolog|fisiolog|fysiolog|fiziolo|fizyolo|физиолог", s) else 0.0
                out[lab] = max(out[lab], v)
        for m in BME.finditer(s):
            v = polarity(s, m.start(), lead_neg)
            if v > 0 and not re.search(r"subchondr|degenera|reactiv|stress|subkondr|subhondr|субхондр", s):
                out["Contusion"] = max(out["Contusion"], 0.5 * v)
    return out


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [label_report(t) for t in df["Report"].tolist()]
    res = pd.DataFrame(rows, columns=LABELS)
    res.insert(0, "StudyInstanceUID", df["StudyInstanceUID"].values)
    return res


def evaluate(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    g = gold.set_index("StudyInstanceUID")[LABELS]
    p = pred.set_index("StudyInstanceUID").loc[g.index, LABELS]
    rows = []
    for k in LABELS:
        y, s = g[k].values.astype(int), p[k].values
        yhat = (s >= 0.5).astype(int)
        tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
        fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
        rows.append(dict(label=k, n_pos=int(y.sum()), acc=(tp + tn) / len(y),
                         sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                         auc=roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan, tp=tp, fp=fp, fn=fn, tn=tn))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home/user/data/meta/train.csv")
    ap.add_argument("--out", default="/home/user/data/meta/silver_labels.csv")
    ap.add_argument("--report", default="results/report_labeler_gold_eval.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.train)
    pred = label_frame(df)
    pred.to_csv(a.out, index=False)
    gold = df[df[LABELS].notna().all(axis=1)]
    ev = evaluate(gold, pred)
    ev.to_csv(a.report, index=False)
    print(ev.round(3).to_string(index=False))
    print("macro AUC vs gold: %.4f  macro acc: %.4f" % (ev.auc.mean(), ev.acc.mean()))
    print("silver prevalence (>=0.5):")
    print((pred[LABELS] >= 0.5).mean().round(3).to_string())
