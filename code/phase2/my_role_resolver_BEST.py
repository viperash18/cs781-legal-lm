import re
PARTY=("APP","RESP")
COUNSEL=("A.COUNSEL","R.COUNSEL")
ROLE4=PARTY+COUNSEL
A_WORDS={"appellant","appellants","petitioner","petitioners","applicant","applicants","plaintiff","plaintiffs","complainant"}
R_WORDS={"respondent","respondents","defendant","defendants","state","union","opposite"}
def side_word(w):                                                    # FIX 3: shared word -> side helper
    w=w.lower()
    if w in A_WORDS: return "A"
    if w in R_WORDS or w.startswith("non"): return "R"
    return None
def toks(s):
    g={"state","union","india","ltd","ors","mr","shri","mrs","ms","dr",
       "the","and","others","another","anr","limited","pvt","smt","sri","government",   # FIX 5: more stop-words
       "appellant","respondent","petitioner"}
    return {w.lower() for w in re.findall(r'[a-zA-Z]+',s) if len(w)>=3 and w.lower() not in g}
def header_bounds(text):
    m=re.search(r'J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|\bJUDGMENT\b|\bO\s*R\s*D\s*E\s*R\b|\bBENCH\s*:|\bCORAM\b',text[:4000])  # FIX 7: case-sensitive, BENCH\s*:
    h_end=m.start() if (m and m.start()>40) else min(1500,len(text))
    v=re.search(r'\b(Vs\.|v\.|Versus)(?![A-Za-z])',text[:h_end],re.I)                     # FIX 2: no \b after "Vs."
    return (h_end,v.start() if v else None)
def header_side(text,s,e,end,div):
    t=text[:end]
    if re.match(r'\W{0,5}(&\s*ors\.?\s*)?\W{0,3}(Vs\.|v\.|Versus)(?![A-Za-z])',t[e:],re.I): return "APP"   # FIX 2: & Ors. + lookahead
    if re.search(r'\b(Vs\.|v\.|Versus)\W{0,6}$',t[:s],re.I): return "RESP"
    w=t[e:e+40]; nv=re.search(r'(Vs\.|v\.|Versus)(?![A-Za-z])',w,re.I); w=w[:nv.start()] if nv else w
    m=re.search(r'\b(Appellant|Petitioner)s?\b(?!\s*:)',w,re.I)
    if m: return "APP"
    m=re.search(r'\b(Respondent)s?\b(?!\s*:)',w,re.I)
    if m: return "RESP"
    m=re.search(r'\b(PETITIONER|APPELLANT)S?\s*:[^:]*$',t[:s],re.I)
    if m: return "APP"
    m=re.search(r'\b(RESPONDENT)S?\s*:[^:]*$',t[:s],re.I)
    if m: return "RESP"
    if div is not None: return "APP" if e<=div else "RESP"
    return None
def counsel_side(text,s,e):
    w=text[e:e+120]                                                                         # FIX 3: window + cut at next label
    nxt=re.search(r'\b(For|FOR|Counsel [Ff]or|Agent [Ff]or)\s',w)
    if nxt: w=w[:nxt.start()]
    m=re.search(r'(appearing for|counsel for|for|on behalf of|representing)\s+(the\s+)?(writ\s+)?([a-z-]+)',w,re.I)
    if m and side_word(m.group(4)): return side_word(m.group(4))
    labs=list(re.finditer(r'\b(For|FOR|Counsel [Ff]or|Agent [Ff]or)\s+(the\s+|The\s+)?(writ\s+|Writ\s+)?(\w+)',text[max(0,s-250):s]))  # FIX 3
    for lab in reversed(labs):
        if side_word(lab.group(4)): return side_word(lab.group(4))
    m=re.search(r'\b(respondent|appellant|petitioner)s?\s+(is|are)\s+represented\s+by\W{0,3}(Mr|Ms|Mrs|Shri|Dr)?\.?\s*$',text[max(0,s-80):s],re.I)  # FIX 4
    if m: return side_word(m.group(1))
    return None
def resolve(text,spans):
    h_end,div=header_bounds(text)
    p_lists={}
    c_map={}
    for sp in spans:
        s,e,l=sp[:3]                                                                       # FIX 1: spans are [start, end, label]
        if l in PARTY and e<=h_end:
            sd=header_side(text,s,e,h_end,div)
            if sd:
                for tk in toks(text[s:e]): p_lists.setdefault(tk,[]).append(sd)
        elif l in COUNSEL:
            sd=counsel_side(text,s,e)
            if sd:
                for tk in toks(text[s:e]): c_map.setdefault(tk,[]).append(sd)
    p_map={k:v[0] for k,v in p_lists.items() if len(set(v))==1}                           # FIX 6: drop conflicting tokens
    fc_map={}
    for k,v in c_map.items():                                                               # FIX 6: strict majority only
        a,r=v.count("A"),v.count("R")
        if a!=r: fc_map[k]="A" if a>r else "R"
    res=[]
    for sp in spans:
        s,e,l=sp[:3]                                                                       # FIX 1
        if l in PARTY:
            sds={p_map[tk] for tk in toks(text[s:e]) if tk in p_map}
            if len(sds)==1: l=sds.pop()
        elif l in COUNSEL:
            sds={fc_map[tk] for tk in toks(text[s:e]) if tk in fc_map}
            if len(sds)==1: l="A.COUNSEL" if sds.pop()=="A" else "R.COUNSEL"
        res.append([s,e,l])                                                                 # FIX 1: [start, end, label]
    return res
