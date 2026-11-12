import os
import re
import spacy
import nltk
from nltk.corpus import wordnet

nlp = spacy.load("en_core_web_sm")

# NLP/code/ -> NLP/
BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IN_DIR = os.path.join(BASE, "input")
COREF_DIR = os.path.join(BASE, "coreferencing_resolution")
OUT_DIR = os.path.join(BASE, "output_ours")
os.makedirs(OUT_DIR, exist_ok=True)


def p(*parts):
    return os.path.join(*parts)


def read_first_line(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readline()


# ---------- 1. Thay dai tu theo ket qua dong tham chieu co san ----------
mapping = {}
with open(p(COREF_DIR, "mapping.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        t = line.rstrip().split("\t")
        if len(t) >= 2:
            mapping[t[0]] = t[1]

coref = {}
with open(p(COREF_DIR, "AllenNLP_coreferencing_resolution_parsed.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        t = line.rstrip().split("\t")
        if len(t) >= 2:
            coref.setdefault(t[0], []).append(t[1])

text = read_first_line(p(IN_DIR, "input_all_beforeCoref.txt"))
for cid, new_term in mapping.items():
    for old in coref.get(cid, []):
        text = text.replace(str(old), new_term)
with open(p(OUT_DIR, "input_all_afterCoref.txt"), "w", encoding="utf-8") as f:
    f.write(text)

# ---------- 2. Tach cau (dung file da co cua tac gia) ----------
text_after = read_first_line(p(IN_DIR, "input_all_afterCoref_new.txt"))
rx = re.compile(r"(,|;|\.|and|then)")
sentences_out = []
for s in rx.split(text_after):
    clean = s.replace(".", "").replace(",", "").replace("and", "").replace("then", "")
    if len(clean.split(" ")) > 2:
        sentences_out.append(s)
with open(p(OUT_DIR, "sentences.txt"), "w", encoding="utf-8") as f:
    for s in sentences_out:
        f.write(s + "\n")

# ---------- 3. Kinds: Actors (nsubj), Processes (verb), Objects (dobj) ----------
actors, verbs, dir_objects = {}, {}, {}
term_id = 0
for sentence in text_after.split("."):
    doc = nlp(sentence)
    for token in doc:
        term_id += 1
        if token.dep_ == "nsubj":
            actors[term_id] = token.lemma_
        elif token.pos_ in ("AUX", "VERB"):
            verbs[term_id] = token.lemma_
        elif token.dep_ == "dobj":
            dir_objects[term_id] = token.lemma_


def dump(name, d):
    with open(p(OUT_DIR, name), "w", encoding="utf-8") as f:
        for k in d:
            f.write(str(k) + "\t" + d[k] + "\n")


dump("actors.txt", actors)
dump("verbs.txt", verbs)
dump("direct_objects.txt", dir_objects)


# ---------- 4. Tasks: bo ba <Actor, Process, Object> ----------
def get_subject_phrase(doc):
    for token in doc:
        if "subj" in token.dep_:
            sub = list(token.subtree)
            return doc[sub[0].i: sub[-1].i + 1]


def get_verb_phrase(doc):
    for token in doc:
        if token.pos_ in ("AUX", "VERB"):
            return token.lemma_


def get_object_phrase(doc):
    for token in doc:
        if token.dep_ in ("dobj", "pobj"):
            sub = list(token.subtree)
            return doc[sub[0].i: sub[-1].i + 1]


triples = []
with open(p(OUT_DIR, "sentences.txt"), "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]
for sentence in lines:
    doc = nlp(sentence)
    sp, vp, op = get_subject_phrase(doc), get_verb_phrase(doc), get_object_phrase(doc)
    if sp is None or vp is None or op is None:
        continue
    for el in sp:
        s = str(el)
        if "customer" in s or "shop" in s or "cook" in s or "delivery" in s:
            triple = "<" + str(sp) + ", TO " + str(vp).upper() + ", " + str(op) + ">"
            triples.append(triple.replace("The", "").replace("the", ""))

with open(p(OUT_DIR, "triples.txt"), "w", encoding="utf-8") as f:
    for t in triples:
        f.write(t + "\n")

# ---------- 5. Glossary tu WordNet ----------
nltk.download("wordnet", quiet=True)
glossary_lines = []
for d in (actors, dir_objects):
    for k in d:
        word = d[k]
        for s in wordnet.synsets(word):
            glossary_lines.append(word + "\t" + s.name() + "\t" + s.definition())
for syn_word, label in (("deliverer", "delivery-boy"), ("website", "web-site")):
    for s in wordnet.synsets(syn_word):
        glossary_lines.append(label + "\t" + s.name() + "\t" + s.definition())

with open(p(OUT_DIR, "glossary_wordnet.txt"), "w", encoding="utf-8") as f:
    for line in glossary_lines:
        f.write(line + "\n")

print("Xong. Ket qua nam trong:", OUT_DIR)
print("So bo ba (Tasks):", len(triples))
for t in triples:
    print(t)
