import json
import argparse
from tqdm import tqdm
import stanza

nlp = stanza.Pipeline('en', processors={'tokenize,ner,pos'})


global_cache = {}

# convert squad format => simple format
"""
convert squad format => simple format

Each element in ['data'] followed by 'title'
{
    "version": "viquad2_training_set",
    "data":
    [
        {
            "title": <string>,
            "paragraphs": [
                {
                    "qas": [
                        {
                            "question": <string>,
                            "answers" / "plausible_answers": [
                                {
                                    "answer_start": <int>, // character level
                                    "text": <string>
                                }
                            ],
                            "id": <string>,
                            "is_impossible": true => "plausible_answers", false => "answers"
                        },
                        ...
                    ],
                    "context": document <string>
                },
                ...
            ]
        },
        {
            "title": "..",
            "paragraphs": [..]
        },
        {
            "title": "..",
            "paragraphs": [..]
        },
        ...
    ]
}

--> simple format:
{
    "title": <string>,
    "context": <string>,
    "question": <string>,
    "answer": <string>,
    "answer_start": <int>, // character level
    "id": <string>
}
"""
def convert2simpleFormat(data): # data = [topics], apply for all triple (c,q,a) instances
    simpleFormatData = []
    for topic in data:
        title = topic['title']
        for para in topic['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                question = qa['question']
                id = qa['id']
                if qa['is_impossible']:
                    try:
                        answer = qa['plausible_answers'][0]['text'] # use only the first answer
                        answer_start = qa['plausible_answers'][0]['answer_start'] # use only the first answer
                    except Exception as e:
                        print("EXCEPTION:\nCONTEXT:", context, "\nQA:", qa)
                else:
                    try:
                        answer = qa['answers'][0]['text'] # use only the first answer
                        answer_start = qa['answers'][0]['answer_start'] # use only the first answer
                    except Exception as e:
                        print("EXCEPTION:\ncontext", context, "\nqa", qa)
                simpleFormatData.append(
                    {
                        "title": title,
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "answer_start": answer_start,
                        "id": id
                    }
                )
    return simpleFormatData


def getAnnotationInfo(simpleFormatData):
    ids = []
    annotation1_toks = []
    annotation1_NERs = []
    annotation1_POSs = []
    annotation1_AnswerStartAndEndPosition = []
    annotation1_AnswerStartCharIndex = []

    annotation2_toks = []
    annotation3_toks = []

    global global_cache

    for instance in tqdm(simpleFormatData):
        # poor performance with stanza
        questions = nlp(instance['question']).to_dict()
        answers = nlp(instance['answer']).to_dict()
        
        # use cache
        if instance['context'] in global_cache:
            contexts = global_cache[instance['context']]
        else:
            global_cache = {} #reset cache
            contexts = nlp(instance['context']).to_dict()
            global_cache[instance['context']] = contexts

        ids.append(instance['id'])
        # question&answer process
        toks_sent = []
        for question in questions:
            for token in question:
                toks_sent.append(token['text'])
        annotation2_toks.append(' '.join(toks_sent))

        toks_sent = []
        for answer in answers:
            for token in answer:
                toks_sent.append(token['text'])
        annotation3_toks.append(' '.join(toks_sent))

        # context process
        toks_sent = []
        NERs_sent = []
        POSs_sent = []
        for context in contexts:
            for token in context:
                toks_sent.append(token['text'])
                NERs_sent.append(token['ner'])
                POSs_sent.append(token['upos'])

        annotation1_toks.append(' '.join(toks_sent))
        annotation1_NERs.append(' '.join(NERs_sent))
        annotation1_POSs.append(' '.join(POSs_sent))
        annotation1_AnswerStartCharIndex.append(instance['answer_start'])

        # annotate graph
    return ids, annotation1_toks, annotation1_NERs, annotation1_POSs, annotation1_AnswerStartAndEndPosition, annotation1_AnswerStartCharIndex, annotation2_toks, annotation3_toks

"""
# convert simple format to squad split format (for using it in Neural Question Generation)
simple format
--> squad split format
{
    "id": string,
    "annotation1": { // context
        "raw_text": string,
        "toks": string,
        "POSs": string,
        "NERs": string,
        "AnswerStartAndEndPosition": "<start> <space> <end>", // calculate in toks, index by word, start from 0
        "AnswerStartCharIndex": "<start>",
        "graph": {
            "g_features": [
                <list of words tokenized in toks> (check the length of this keyword that must equal to len(['toks'].split(' '))
            ],
            "g_adj": {
                <in progress>: ⏸ ⏸ ⏸
            },
            "num_edges": 30
        }
    },
    "annotation2": { // question
        "raw_text": string,
        "toks": string, // tokenized sentence
    },
    "annotation3": { // answer
        "raw_text": string,
        "toks": string, // tokenized sentence
    },
}
"""
def convert2squadsplitFormat(simpleFormatData): # simpleFormatData = [topics], apply for all triplet instances
    squadSplitFormatData = []
    ids, annotation1_toks, annotation1_NERs, annotation1_POSs, annotation1_AnswerStartAndEndPosition, annotation1_AnswerStartCharIndex, annotation1_graph, annotation2_toks, annotation3_toks = getAnnotationInfo(simpleFormatData)

    for i in range(len(ids)):
        squadSplitFormatData.append(
            {
                "id": ids[i],
                "annotation1": {
                    "raw_text": simpleFormatData[i]['context'],
                    "toks": annotation1_toks[i],
                    "POSs": annotation1_POSs[i],
                    "NERs": annotation1_NERs[i],
                    "AnswerStartAndEndPosition": "<start> <space> <end>",
                    "AnswerStartCharIndex": annotation1_AnswerStartCharIndex[i],
                    "graph": annotation1_graph[i]
                },
                "annotation2": {
                    "raw_text": simpleFormatData[i]['question'],
                    "toks": annotation2_toks[i],
                },
                "annotation3": {
                    "raw_text": simpleFormatData[i]['answer'],
                    "toks": annotation3_toks[i],
                }
            }
        )

    return squadSplitFormatData



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the squad raw file file')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to the output file')

    args = vars(parser.parse_args())

    """
    convert from "raw_format" --> simple format --> squad split format
    """

    with open(args['input'], 'r') as f:
        raws = json.load(f)
        final_datas = convert2squadsplitFormat(convert2simpleFormat(raws))

    with open(args['output'], 'w') as f:
        json.dump(final_datas, f)

    pass
