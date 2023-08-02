import json
import argparse
from collections import defaultdict
import networkx as nx
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import itertools
import re

model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)  # load the model

global_cache = {}


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # words = text.split(' ')
        # words = re.findall(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]|\w+""", text)
        global global_cache

        if text not in global_cache:
            global_cache = {} # reset cache
            prediction = predictor.predict(document=text)
            global_cache[text] = prediction
        else:
            prediction = global_cache[text]

        words = prediction['document']

        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class WhitespaceRegexTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # words = text.split(' ')
        words = re.findall(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]|\w+""", text)

        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

regex_nlp = spacy.load('en_core_web_sm')
regex_nlp.tokenizer = WhitespaceRegexTokenizer(nlp.vocab)

def get_coref_span_head(parser, span):
    doc_mention = parser(span)
    for sent in doc_mention.sents:
        for token in sent:
            if token.text == token.head.text and token.i == token.head.i:
                return token.i


def get_head_of_sentence(sentence):
    for token in sentence:
        if token.text == token.head.text and token.i == token.head.i:
            return token.i

def extract_sent_coref_dep_tree(parser, text):
    global global_cache
    if len(text) == 0:
        return {'g_features': [], 'g_adj': {}, 'num_edges': 0}

    if text not in global_cache:
        global_cache = {} # reset cache
        prediction = predictor.predict(document=text)
        global_cache[text] = prediction
    else:
        prediction = global_cache[text]
    coref_clusters = prediction['clusters']
    document_tokens = prediction['document']
    span_connection = [] # tuple(mention_head, span_head)
    mention_heads = []
    # trace_mention = {}
    # trace_head = {}
    for clst in coref_clusters:
        span = clst[0]
        mentions = clst[1:]
        for i in range(len(mentions)):
            mentions[i] = range(mentions[i][0], mentions[i][1]+1) # fill [start, end] by [start, start+1, start+2, ..., end-1, end] (full mention)
            if len(mentions[i]) == 1:
                mention_heads.append(mentions[i][0])
                # trace_mention[mention_heads[-1]] = document_tokens[mentions[i][0]]
            else:
                mention_text = ' '.join([document_tokens[j] for j in mentions[i]])
                mention_heads.append(mentions[i][0] + get_coref_span_head(regex_nlp, mention_text))
                # trace_mention[mention_heads[-1]] = mention_text
        # span[0] + .. ==> get index at origin text
        ' '.join(document_tokens[span[0]:span[1]+1])
        if span[0] == span[1]:
            head = span[0]
            # trace_head[head] = document_tokens[head]
        else:
            head = span[0] + get_coref_span_head(regex_nlp, ' '.join(document_tokens[span[0]:span[1]+1]))
            # trace_head[head] = document_tokens[head]
        for mention in mention_heads:
            span_connection.append((mention, head))
        
        
    doc = parser(text)
    boundary_nodes = []

    # displacy.render(list(doc.sents), style="dep", jupyter=True, options={'distance':140}) # visualize in Jupyter Notebook
    g_features = []
    dep_tree = defaultdict(list)
    num_edges = 0
    for sent in doc.sents:
        boundary_nodes.append(get_head_of_sentence(sent))
        for each in sent:
            g_features.append(each.text)
            if each.i != each.head.i: # Not a root
                dep_tree[each.head.i].append({'node': each.i, 'edge': each.dep_})
                num_edges += 1

    for i in range(len(boundary_nodes) - 1):
        # Add connection between neighboring dependency trees
        dep_tree[boundary_nodes[i]].append({'node': boundary_nodes[i+1], 'edge': 'neigh'})
        dep_tree[boundary_nodes[i+1]].append({'node': boundary_nodes[i], 'edge': 'neigh'})
        num_edges += 2

    for tup in span_connection:
        # if tup[0] > len(g_features) or tup[1] > len(g_features):
        #     print(tup)
        #     print(trace_mention[tup[0]], trace_mention[tup[1]])
        dep_tree[tup[0]].append({'node': tup[1], 'edge': 'coref'})
        # dep_tree[tup[1]].append({'node': tup[0], 'edge': 'coref'})
        num_edges += 1

    info = {'g_features': g_features,
            'g_adj': dep_tree,
            'num_edges': num_edges
            }
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input file')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to the output file')
    parser.add_argument('-start', '--start', required=False, default=0, type=int)
    parser.add_argument('-end', '--end', required=False, type=int)

    args = vars(parser.parse_args())

    with open(args['input'], encoding='utf-8') as dataset_file:
        dataset = json.load(dataset_file)

        all_instances = []
        print(len(dataset[args['start']:args['end']]))
        for instance in tqdm(dataset[args['start']:args['end']]):
            tokens = re.split(' +', instance['annotation1']['toks'])
            # tokens = re.findall(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]|\w+""", instance['annotation1']['toks'])
            graph = extract_sent_coref_dep_tree(nlp, ' '.join(tokens))
            if len(tokens) != len(graph['g_features']):
                print(tokens)
                print(graph['g_features'])
                raise Exception("Sorry toi chiu") 
                # assert False
            instance['annotation1']['graph'] = graph
            all_instances.append(instance)

        with open(args['output'], 'w') as out_file:
            json.dump(all_instances, out_file)

