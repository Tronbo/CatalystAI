"""Idea taken from https://www.notion.so/Analogies-Generator-9b046963f52f446b9bef84aa4e416a4c"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from api import GPT, Example, UIConfig, set_openai_key
from api import demo_web_app
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

from flask import Flask, render_template
from flask import request

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

set_openai_key("sk-MXAmsogskTEQLCKH80Mx3rbqsQyTsJxFnHlOfy1h")

candidate_sentences = pd.read_csv("data.csv")
print(candidate_sentences.shape)

# Construct GPT object and show some examples
gpt = GPT(engine="davinci",
            temperature=0.7,
            max_tokens=300)

gpt.add_example(Example('Harmonization/domain-invariance schemes results are undesirable. The accuracy of the output predictions cannot be accurately predicted.',
                            'We show that for a wide class of harmonization/domain-invariance schemes several undesirable properties are unavoidable. If a predictive machine is made invariant to a set of domains, the accuracy of the output predictions (as measured by mutual information) is limited by the domain with the least amount of information to begin with. If a real label value is highly informative about the source domain, it cannot be accurately predicted by an invariant predictor. These results are simple and intuitive, but we believe that it is beneficial to state them for medical imaging harmonization.'))
gpt.add_example(Example('Prediction in a new domain without any training sample called zero-shot domain adaptation (ZSDA), is an important task in domain adaptation.',
                            'Prediction in a new domain without any training sample, called zero-shot domain adaptation (ZSDA), is an important task in domain adaptation. While prediction in a new domain has gained much attention in recent years, in this paper, we investigate another potential of ZSDA. Specifically, instead of predicting responses in a new domain, we find a description of a new domain given a prediction. The task is regarded as predictive optimization, but existing predictive optimization methods have not been extended to handling multiple domains. We propose a simple framework for predictive optimization with ZSDA and analyze the condition in which the optimization problem becomes convex optimization. We also discuss how to handle the interaction of characteristics of a domain in predictive optimization. Through numerical experiments, we demonstrate the potential usefulness of our proposed framework.'))
gpt.add_example(Example('Agents that learn to select optimal actions represent an important focus of the sequential decision making literature.',
                            'Agents that learn to select optimal actions represent a prominent focus of the sequential decisionmaking literature. In the face of a complex environment or constraints on time and resources, however, aiming to synthesize such an optimal policy can become infeasible. These scenarios give rise to an important trade-off between the information an agent must acquire to learn and the sub-optimality of the resulting policy. While an agent designer has a preference for how this trade-off is resolved, existing approaches further require that the designer translate these preferences into a fixed learning target for the agent. In this work, leveraging rate-distortion theory, we automate this process such that the designer need only express their preferences via a single hyperparameter and the agent is endowed with the ability to compute its own learning targets that best achieve the desired trade-off. We establish a general bound on expected discounted regret for an agent that decides what to learn in this manner along with computational experiments that illustrate the expressiveness of designer preferences and even show improvements over Thompson sampling in identifying an optimal policy.'))
gpt.add_example(Example('We study learning Censor Markov Random Fields CMRF in short. These are Markov Random Fields where some of the nodes are censored.',
                            'We study learning Censor Markov Random Fields (abbreviated CMRFs). These are Markov Random Fields where some of the nodes are censored (not observed). We present an algorithm for learning high temperature CMRFs within o(n) transportation distance. Crucially our algorithm makes no assumption about the structure of the graph or the number or location of the observed nodes. We obtain stronger results for high girth high temperature CMRFs as well as computational lower bounds indicating that our results can not be qualitatively improved.'))
gpt.add_example(Example('Federated learning brings potential benefits of faster learning, better solutions, and a greater propensity to transfer when heterogeneous data from different parties increases diversity.',
                            'Federated learning brings potential benefits of faster learning, better solutions, and a greater propensity to transfer when heterogeneous data from different parties increases diversity. However, because federated learning tasks tend to be large and complex, and training times non-negligible, it is important for the aggregation algorithm to be robust to non-IID data and corrupted parties. This robustness relies on the ability to identify, and appropriately weight, incompatible parties. Recent work assumes that a reference dataset is available through which to perform the identification. We consider settings where no such reference dataset is available; rather, the quality and suitability of the parties needs to be inferred. We do so by bringing ideas from crowdsourced predictions and collaborative filtering, where one must infer an unknown ground truth given proposals from participants with unknown quality. We propose novel federated learning aggregation algorithms based on Bayesian inference that adapt to the quality of the parties. Empirically, we show that the algorithms outperform standard and robust aggregation in federated learning on both synthetic and real data.'))
    

def get_relation(sent):
    doc = nlp(sent)

    matcher = Matcher(nlp.vocab)

    pattern = [{'DEP':'ROOT'}, {'DEP':'prep','OP':"?"}, {'DEP':'agent','OP':"?"}, {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", None, pattern) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)


def get_entities(sent):
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""
    prv_tok_text = ""

    prefix = ""
    modifier = ""
    
    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text

        if tok.dep_.endswith("mod") == True:
            modifier = tok.text
            if prv_tok_dep == "compound":
                modifier = prv_tok_text + " "+ tok.text
        
        if tok.dep_.find("subj") == True:
            ent1 = modifier +" "+ prefix + " "+ tok.text
            prefix = ""
            modifier = ""
            prv_tok_dep = ""
            prv_tok_text = ""      

        if tok.dep_.find("obj") == True:
            ent2 = modifier +" "+ prefix +" "+ tok.text

        prv_tok_dep = tok.dep_
        prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]

entity_pairs = []
relations = []

'''
for i in tqdm(candidate_sentences):
    individual_sentences = i.split(". ")
    for j in individual_sentences:
        entity_pairs.append(get_entities(j))


for i in tqdm(candidate_sentences):
    individual_sentences = i.split(". ")
    for j in individual_sentences:
        try:
            relations += [get_relation(j)]
        except:
            relations += [" "]

print(relations)

source = [i[0] for i in entity_pairs]

target = [i[1] for i in entity_pairs]

print(source)
print(target)

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

G=nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

print(len(source))

plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color="gray", edge_color="skyblue", edge_cmap=plt.cm.Blues, pos = pos)
'''

def compute(string):
    new_arr = string.split(" ")
    final_arr = []

    for i in new_arr:
        try:
            curr_new_word = target[source.index(i)]
            final_arr += [i]
            final_arr += ["(" + curr_new_word + ")"]
        except:
            final_arr += [i]
    final_string = ""

    for i in final_arr:
        final_string += i + " "

    print(final_string)

    main_Data = gpt.get_top_reply(final_string)
    return main_Data

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/demo')
def demo():
    return render_template("demo.html")

@app.route('/demo', methods=['POST'])
def my_form_post():
    text = request.form['text']
    essay = "O" + compute(text)[1:]
    return render_template("demo.html", output=essay)

app.run()