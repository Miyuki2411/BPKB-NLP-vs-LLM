#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_md")


# In[2]:


path="../EKGwithNLP"


# #### Example of input

# In[3]:


example = "On the arrival of Mary’s order at PizzaPazza, John, the cook, puts the order on the worklist."
doc = nlp(example)
for token in doc:
    print(token.lemma_+"\tPOS: "+token.pos_+"\tDEP: " + token.dep_)


# #### Function for extracting noun chunks

# In[4]:


def noun_chunks(doc):
    print("chunk analysis\n")
    num_chunks = 0
    for chunk in doc.noun_chunks:
        num_chunks = num_chunks + 1
        print("chunk text: "+chunk.text+", root text: "+chunk.root.text+", root dep: "+ chunk.root.dep_+", root head: "+chunk.root.head.text)
    num_chunks = len(list(doc.noun_chunks))
    print("total number of chunks: "+str(num_chunks))


# In[5]:


example = "Mary, the customer, connects to the PizzaPazza web-site and places her order of two Napoli pizzas, providing also the payment."
doc = nlp(example)
noun_chunks(doc)  
    


# In[6]:


with open (path+"input/input_all_beforeCoref.txt", "r") as fin:
    text = fin.readline()

    text_sentences = text.split(".")

    terms = {}
    term_id = 0
    
    for sentence in text_sentences:
        #print(sentence)
        doc = nlp(sentence)
        
        for token in doc:
            term_id = term_id +1
            terms[term_id] = token.text


# In[7]:


with open(path+"output/terms_beforeCoref.txt", "w+") as fout:
    for k in terms:
        fout.write(str(k)+"\t"+terms[k]+"\n")


# #### Co-referencing resolution

# With Allen NLP
# https://demo.allennlp.org/coreference-resolution/

# In[ ]:


get_ipython().system(' python3 -m pip install --user allennlp')
get_ipython().system(' python3 -m pip install --user allennlp_models')


# In[ ]:


from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
# predictor.predict(
#     document="Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers."
# )
predictor.predict(document=text)


# In[ ]:


text


# https://demo.allennlp.org/coreference-resolution/s/my-business-pizzapazza-is-home/I5Y6Z3N7U9

# In[8]:


# replace nouns, pronouns, and personal adjectives with the terms results of the mapping given by the conferencing resolution with AllenNLP

mapping_pronouns_terms = {}
coreferences = {}

with open(path+"coreferencing_resolution/mapping.tsv", "r") as fin:
    for line in fin.readlines():
        tokens = line.rstrip().split("\t")
        id = tokens[0]
        term = tokens[1]
        mapping_pronouns_terms[id] = term
print(str(mapping_pronouns_terms))

with open(path+"coreferencing_resolution/AllenNLP_coreferencing_resolution_parsed.tsv", "r") as fin:
    for line in fin.readlines():
        tokens = line.rstrip().split("\t")
        id = tokens[0]
        terms = tokens[1]
        if id in coreferences:
            list_of_terms_to_replace = coreferences[id]
            list_of_terms_to_replace.append(terms)            
        else:
            list_of_terms_to_replace = []
            list_of_terms_to_replace.append(terms)
            
        coreferences[id] = list_of_terms_to_replace
print(str(coreferences))

with open(path+"input/input_all_afterCoref.txt", "w+") as fout:
    with open(path+"input/input_all_beforeCoref.txt", "r") as fin:
        text = fin.readline()
        for el in mapping_pronouns_terms:
            new_terms = mapping_pronouns_terms[el]            
            if el in coreferences:
                terms_to_replace = coreferences[el]
                for t in terms_to_replace:
                    text = text.replace(str(t), new_terms)
    fout.write(text)


# In[9]:


#splitting text into sentences 
import re

with open(path+"input/sentences.txt", "w+") as fout:
    with open (path+"input/input_all_afterCoref_new.txt", "r") as fin:
        text = fin.readline()
        rx = re.compile(r'(,|;|\.|and|then)') #(use full stop, semicolon, comma, and conjunctions)
        text_sentences=rx.split(text)
        #text_sentences = re.split('\.|;|,', text)
        for text_sentence in text_sentences:
            #take only relevant ones, more than 2 words
            text_sentence_clean = text_sentence.replace(".", "").replace(",", "").replace("and", "").replace("then", "")
            if len(text_sentence_clean.split(" ")) > 2:
                fout.write(text_sentence+"\n")


# #### Categorization

# In[10]:


#categorization: find actors (nsubj), verbs, objects (pobj, dobj)

with open (path+"input/input_all_afterCoref_new.txt", "r") as fin:
    text = fin.readline()

    text_sentences = text.split(".")

    terms_new = {} #pronouns and personal adj are replaced with the mapping given by the co-referencing resolution step
    actors = {}
    verbs = {}
    objects = {}
    dir_objects = {}
    term_id = 0

    for sentence in text_sentences:
        print(sentence)
        doc = nlp(sentence)

        for token in doc:
            term_id = term_id +1
            terms_new[term_id] = token.text
            print(token.lemma_+"\tPOS: "+token.pos_+"\tDEP: " + token.dep_)
            if(token.dep_ == "nsubj"):
                print("Actor found")
                print(str(term_id)+" "+token.lemma_+"\tPOS: "+token.pos_+"\tDEP: " + token.dep_)
                actors[term_id] = token.lemma_
            elif (token.pos_ == "AUX" or token.pos_ == "VERB"):
                print("Verb found")
                print(str(term_id)+" "+token.lemma_+"\tPOS: "+token.pos_+"\tDEP: " + token.dep_)
                verbs[term_id] = token.lemma_
            elif(token.dep_ == "dobj"):
                print("Object found")
                print(str(term_id)+" "+token.lemma_+"\tPOS: "+token.pos_+"\tDEP: " + token.dep_)
                dir_objects[term_id] = token.lemma_
            #if(token.dep_ == "dobj" or token.dep_ == "pobj"):
            #    print("Object found")
            #    print(str(term_id)+" "+token.lemma_+"\tPOS: "+token.pos_+"\tDEP: " + token.dep_)
            #    objects[term_id] = token.lemma_


# In[11]:


with open(path+"output/terms.txt", "w+") as fout:
    for k in terms_new:
        fout.write(str(k)+"\t"+terms_new[k]+"\n")


# In[12]:


with open(path+"output/actors.txt", "w+") as fout:
    for k in actors:
        fout.write(str(k)+"\t"+actors[k]+"\n")


# In[13]:


with open(path+"output/verbs.txt", "w+") as fout:
    for k in verbs:
        fout.write(str(k)+"\t"+verbs[k]+"\n")


# In[14]:


with open(path+"output/objects.txt", "w+") as fout:
    for k in objects:
        fout.write(str(k)+"\t"+objects[k]+"\n")


# In[15]:


with open(path+"output/direct_objects.txt", "w+") as fout:
    for k in dir_objects:
        fout.write(str(k)+"\t"+dir_objects[k]+"\n")


# #### Tasks

# In[16]:


#for triples: check if there is a subj, a verb, and obj in a sequence, take only complete triples with an actor performing an action

triples = []
def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]
        
def get_verb_phrase(doc):
    verbs_original_form = ""
    for token in doc:
        if ("AUX" in token.pos_) or ("VERB" in token.pos_) :
            return token.lemma_
        #    verbs_original_form = verbs_original_form +" "+ token.lemma_
         #return verbs_original_form

def get_object_phrase(doc):
    for token in doc:
        if ("dobj" in token.dep_) or ("pobj" in token.dep_) :
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

subject_phrase = "" 
verb_phrase = "" 
object_phrase  = ""   
        
with open (path+"input/sentences.txt", "r") as fin:
    lines = [line.rstrip() for line in fin]
    for sentence in lines:
        doc = nlp(sentence)
        subject_phrase = get_subject_phrase(doc)
        verb_phrase = get_verb_phrase(doc)
        object_phrase = get_object_phrase(doc)
        if subject_phrase == None or verb_phrase == None or object_phrase == None: #not complete triple
            continue
    
        #select triples with an actor
        for el in subject_phrase:
            if "customer" in str(el) or "shop" in str(el) or "cook" in str(el) or "delivery" in str(el):
                triple= "<"+str(subject_phrase)+", TO "+str(verb_phrase).upper()+", "+str(object_phrase)+">"
                triple_cleaned = triple.replace("The", "").replace("the","")
                
                triples.append(triple_cleaned)
                
with open(path+"output/triples.txt", "w+") as fout:
    for t in triples:
        print(t)
        fout.write(str(t)+"\n")


# #### Glossary

# In[17]:


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()


# In[18]:


import nltk

nltk.download('wordnet')
  


# In[19]:


from nltk.corpus import wordnet

glossary_actors ={}

for a in actors:
    actor = actors[a]   
    for words in wordnet.synsets(actor): 
        #actor_id = str(a)+":"+actor
        #glossary_actors[actor_id] = words.definition()
        print(actor+"\t"+words.name()+"\t"+words.definition())
        


# In[20]:


from nltk.corpus import wordnet

glossary_objs ={}

for o in dir_objects:
    obj = dir_objects[o]   
    for words in wordnet.synsets(obj): 
        #obj_id = str(o)+":"+obj
        #glossary_objs[obj] = words.definition()
        print(obj+"\t"+words.name()+"\t"+words.definition())
        


# In[21]:


for words in wordnet.synsets("deliverer"): 
    print("delivery-boy\t"+words.name()+"\t"+words.definition())


# In[22]:


for words in wordnet.synsets("website"): 
    print("web-site\t"+words.name()+"\t"+words.definition())


# In[ ]:




