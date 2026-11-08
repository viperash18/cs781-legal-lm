"""Document-level role resolver for L-NER predictions (post-processing; no model, no training).

Every party / counsel name keeps ONE role within a judgment (consistency 1.000 in fold1+fold2 gold),
and the role is stated explicitly in the text. The model labels mentions one by one, so it often
flips APP<->RESP and A.COUNSEL<->R.COUNSEL. This script re-derives the side from explicit cues:

  parties  (APP/RESP), header only:
    1. "X Vs. Y"                       -> X appellant side, Y respondent side
    2. "X ..Appellant(s)" / "Y ..Respondent(s)"   suffix labels
    3. "PETITIONER: X"  /  "RESPONDENT: Y"        prefix labels
    4. otherwise: before / after the first "Versus" of the header
  counsel  (A.COUNSEL/R.COUNSEL), anywhere:
    1. "Mr X, learned counsel appearing for the appellant/respondent"
    2. "For the Appellant: Mr X" / "Counsel for Respondent :- Y"   (label before the name)
    3. "The respondent is represented by Mr X"
Each name's side is decided once per document (majority over cues) and then applied to every
mention whose distinctive tokens point to exactly one side. Only the role label changes; spans
and all other labels are untouched.  Designed with help from Claude (declared in the report).
"""
import re, json, collections
PARTY = ("APP", "RESP"); COUNSEL = ("A.COUNSEL", "R.COUNSEL"); ROLE4 = PARTY + COUNSEL
STOP = set("""state of the and ors anr others another union india government govt ltd limited pvt private
co company m s mr mrs smt shri sri dr ms through its by l r lrs secretary director managing general
municipal corporation council board commissioner officer district bank insurance national new
union-of- appellant respondent petitioner""".split())
HDR_END = re.compile(r"J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|\bJUDGMENT\b|\bO\s*R\s*D\s*E\s*R\b|\bDATE OF JUDGMENT|\bBENCH\s*:|\bCORAM\b|\bCORAM\s*:", re.I)
VS = re.compile(r"(?<![A-Za-z])(versus|vs\.?|v\.)(?![A-Za-z])", re.I)
A_W = r"(appellant|petitioner|applicant|plaintiff|complainant|revisionist)"
R_W = r"(respondent|opposite part|defendant|state|non-applicant)"

def toks(s):
    return [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP]

def header_bounds(text):
    m = HDR_END.search(text, 0, 4000)
    end = m.start() if m and m.start() > 40 else min(len(text), 1500)
    v = VS.search(text, 0, end)
    rmark = re.search(r"\bRESPONDENTS?\s*:", text[:end])
    div = rmark.start() if rmark else (v.start() if v else None)
    return end, div

SUF_A = re.compile(r"^[\s.\-–…&,()A-Za-z]{0,40}?\b(appellants?|petitioners?|applicants?|plaintiffs?)(\(s\))?\b(?!\s*:)", re.I)
SUF_R = re.compile(r"^[\s.\-–…&,()A-Za-z]{0,40}?\b(respondents?|defendants?|opposite part(y|ies))(\(s\))?\b(?!\s*:)", re.I)
PRE = re.compile(r"\b(PETITIONERS?|APPELLANTS?|RESPONDENTS?)\s*:", re.I)

def header_side(text, s, e, end, div):
    after = text[e:e+60]; before = text[max(0, s-20):s]
    if re.match(r"^\W{0,5}(&\s*(ors|anr)\.?|and (others|another))?\W{0,5}(versus|vs\.?|v\.)(?![a-z])", after, re.I): return "APP"
    if re.search(r"(versus|vs\.?|v\.)\W{0,6}$", before, re.I): return "RESP"
    nv = VS.search(after); cut = after[:nv.start()] if nv else after
    ma, mr = SUF_A.match(cut), SUF_R.match(cut)
    if ma and (not mr or ma.end() < mr.end()): return "APP"
    if mr: return "RESP"
    pre = list(PRE.finditer(text, max(0, s-250), s))
    if pre: return "RESP" if pre[-1].group(1).upper().startswith("RESP") else "APP"
    if div is not None: return "APP" if e <= div else "RESP"
    return None

def party_map(text, spans):
    """Header rules: Vs. adjacency, then '..Appellant(s)' / '..Respondent(s)' suffix, then
    'PETITIONER:' / 'RESPONDENT:' prefix, then position relative to the first Versus."""
    end, div = header_bounds(text)
    roles = {}
    for s, e, l in spans:
        if l in PARTY + ("WIT",) and s < end:
            side = header_side(text, s, e, end, div)
            if side is None: continue
            for t in toks(text[s:e]): roles.setdefault(t, collections.Counter())[side] += 1
    return {t: c.most_common(1)[0][0] for t, c in roles.items() if len(c) == 1}, end

LABEL = re.compile(r"\b(For|FOR|Counsel [Ff]or|COUNSEL FOR|Agent [Ff]or)\s+(the\s+|The\s+)?(writ\s+|Writ\s+)?(\w+)")
def _side_word(w):
    w = w.lower()
    if w.startswith("non") or w.startswith("opposite"): return "R"
    if re.match(A_W, w): return "A"
    if re.match(R_W, w) or w in ("union", "respondents", "state"): return "R"
    return None

def counsel_side(text, s, e):
    # (1) right after the name: "..., learned counsel appearing for the appellants"  (stop at the next 'For X' label)
    after = text[e:e+120]
    nxt = LABEL.search(after); after = after[:nxt.start()] if nxt else after
    m = re.search(r"\b(appearing for|counsel for|for|on behalf of|representing)\s+(the\s+)?(writ\s+)?(\w+)", after.lower())
    if m and _side_word(m.group(4)): return _side_word(m.group(4))
    m = re.search(r"\.{2,}\s*for\s+the\s+(\w+)", text[e:e+160].lower())          # list style "... ...for the petitioner"
    if m and _side_word(m.group(1)): return _side_word(m.group(1))
    # (2) nearest capitalised label before the name: "For the Appellant: Mr X", "Counsel for Respondent :- Y"
    labs = list(LABEL.finditer(text, max(0, s-250), s))
    for lab in reversed(labs):
        sw = _side_word(lab.group(4))
        if sw: return sw
    # (3) "The respondent is represented by Mr. X"
    m = re.search(r"\b(\w+)s?\s+(is|are)\s+represented\s+by\W{0,5}(mr|ms|mrs|shri|dr)?\.?\s*$", text[max(0, s-70):s].lower())
    if m and _side_word(m.group(1)): return _side_word(m.group(1))
    return None

def resolve(text, spans):
    """spans: list of [s,e,label]. Returns relabelled copy (only role labels change)."""
    spans = [list(x[:3]) for x in spans]
    pmap, hend = party_map(text, spans)
    # counsel: document-level vote per surname-token
    cvote = collections.defaultdict(collections.Counter)
    for s, e, l in spans:
        if l in COUNSEL:
            side = counsel_side(text, s, e)
            if side:
                for t in toks(text[s:e]): cvote[t][side] += 1
    cmap = {t: c.most_common(1)[0][0] for t, c in cvote.items() if c.most_common(1)[0][1] > sum(c.values()) - c.most_common(1)[0][1]}
    out = []
    for s, e, l in spans:
        tk = toks(text[s:e])
        if l in PARTY and tk:
            sides = {pmap[t] for t in tk if t in pmap}
            if len(sides) == 1: l = sides.pop()
        elif l in COUNSEL and tk:
            sides = {cmap[t] for t in tk if t in cmap}
            if len(sides) == 1: l = "A.COUNSEL" if sides.pop() == "A" else "R.COUNSEL"
        out.append([s, e, l])
    return out



# ======================================================================================
# usage (run in the folder that contains the unzipped lner_data files):
#   python3 fix_roles.py validate                         -> checks the rules against fold1+fold2 GOLD
#   python3 fix_roles.py apply sub9.json                  -> writes fixed/prediction.json + submission_roles.zip
# ======================================================================================
if __name__ == "__main__":
    import sys, random, zipfile, os
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        T, G = {}, {}
        for f in ("fold1", "fold2"):
            T.update(json.load(open(f"{f}-text.json"))); G.update(json.load(open(f"{f}-gold.json")))
        for p in (0.0, 0.5):     # p = share of role labels flipped to the wrong side before resolving
            rnd = random.Random(0); st = collections.Counter()
            for d, g in G.items():
                gold = [(s, e, l) for s, e, l, *_ in g]; noisy = []
                for s, e, l in gold:
                    if l in PARTY and rnd.random() < p: l = "RESP" if l == "APP" else "APP"
                    elif l in COUNSEL and rnd.random() < p: l = "R.COUNSEL" if l == "A.COUNSEL" else "A.COUNSEL"
                    noisy.append([s, e, l])
                fixed = resolve(T[d], noisy)
                for (s, e, lg), (_, _, ln), (_, _, lf) in zip(gold, noisy, fixed):
                    if lg in ROLE4: st[(lg, "b")] += ln == lg; st[(lg, "a")] += lf == lg; st[(lg, "n")] += 1
            print(f"--- {int(p*100)}% of role labels flipped before resolving")
            for l in ROLE4:
                n = st[(l, "n")]; print(f"  {l:10s} n={n:4d} | role accuracy {st[(l,'b')]/n:.3f} -> {st[(l,'a')]/n:.3f}")
    else:
        pred = json.load(open(sys.argv[2])); t3 = json.load(open("fold3-text.json"))
        out, chg = {}, collections.Counter()
        for d, sp in pred.items():
            out[d] = resolve(t3[d], sp) if sp else sp
            for a, b in zip(sp, out[d]):
                if a[2] != b[2]: chg[(a[2], b[2])] += 1
        os.makedirs("fixed", exist_ok=True)
        json.dump(out, open("fixed/prediction.json", "w"))
        with zipfile.ZipFile("submission_roles.zip", "w") as z:
            z.write("fixed/prediction.json", arcname="prediction.json")
        print("docs", len(out), "| entities", sum(len(v) for v in out.values()), "| relabelled", dict(chg))
        print("wrote fixed/prediction.json and submission_roles.zip")
