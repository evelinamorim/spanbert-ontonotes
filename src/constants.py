# original from:https://github.com/WenzhengZhang/Seq2seqCoref/tree/main
from transformers import T5Tokenizer
from copy import deepcopy


int_tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=4096)

SPEAKER_START = '<speaker>'
SPEAKER_END = '</speaker>'
MENTION_START = '<m>'
MENTION_END = '</m>'
SEP_TOKEN = '|'
COPY = '<copy>'

int_tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                          MENTION_START, MENTION_END, COPY])
SPECIAL_IDS = {
    'speaker_start': int_tokenizer.encode(SPEAKER_START,
                                          add_special_tokens=False)[0],
    'speaker_end': int_tokenizer.encode(SPEAKER_END, add_special_tokens=False)[
        0],
    'mention_start': int_tokenizer.encode(MENTION_START,
                                          add_special_tokens=False)[0],
    'mention_end': int_tokenizer.encode(MENTION_END, add_special_tokens=False)[
        0],
    'sep': int_tokenizer.encode(SEP_TOKEN, add_special_tokens=False)[0],
    'copy': int_tokenizer.encode(COPY, add_special_tokens=False)[0],
    'eos': int_tokenizer.eos_token_id
}
integers = []
for i in range(500):
    cid = int_tokenizer.encode(str(i), add_special_tokens=False)
    integers.extend(cid)
integers = list(set(integers))
SPECIAL_IDS['integers'] = integers

mention_end_non_int_tokenizer = T5Tokenizer.from_pretrained(
    "t5-small",
    model_max_length=4096)
CLUSTER_NEW = '</new>'
CLUSTERS = []
for i in range(500):
    c = f'<c{i}>'
    CLUSTERS.append(c)
mention_end_non_int_tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                                          MENTION_START, MENTION_END, COPY,
                                          CLUSTER_NEW] +
                                         CLUSTERS)
CLUSTER_IDS = [mention_end_non_int_tokenizer.encode(
    e,
    add_special_tokens=False)[0] for e in CLUSTERS]
CLUSTER_TO_NUM = {e: i for i, e in enumerate(CLUSTERS)}
CLUSTER_IDS_TO_NUM = {e: i for i, e in enumerate(CLUSTER_IDS)}
MENTION_END_NON_INT_SPECIAL_IDS = {
    'speaker_start': mention_end_non_int_tokenizer.encode(
        SPEAKER_START,
        add_special_tokens=False)[0],
    'speaker_end':
        mention_end_non_int_tokenizer.encode(
            SPEAKER_END, add_special_tokens=False)[0],
    'mention_start': mention_end_non_int_tokenizer.encode(
        MENTION_START,
        add_special_tokens=False)[0],
    'mention_end': mention_end_non_int_tokenizer.encode(
        MENTION_END,
        add_special_tokens=False)[0],
    'cluster_ids': CLUSTER_IDS,
    'cluster_ids_to_num': CLUSTER_IDS_TO_NUM,
    'cluster_new': mention_end_non_int_tokenizer.encode(
        CLUSTER_NEW,
        add_special_tokens=False)[0],
    'copy': mention_end_non_int_tokenizer.encode(
        COPY, add_special_tokens=False)[0],
    'eos': mention_end_non_int_tokenizer.eos_token_id
}

mark_sent_tokenizer = T5Tokenizer.from_pretrained("t5-small",
                                                  model_max_length=4096)

SENTENCE_START = '<sentence>'
SENTENCE_END = '</sentence>'

mark_sent_tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                                MENTION_START, MENTION_END, COPY,
                                SENTENCE_START,
                                SENTENCE_END])
MARK_SPECIAL_IDS = deepcopy(SPECIAL_IDS)
MARK_SPECIAL_IDS['sentence_start'] = mark_sent_tokenizer.encode(
    SENTENCE_START,
    add_special_tokens=False)[0]
MARK_SPECIAL_IDS['sentence_end'] = mark_sent_tokenizer.encode(
    SENTENCE_END, add_special_tokens=False)[0]

non_int_tokenizer = T5Tokenizer.from_pretrained(
    "t5-small",
    model_max_length=4096)
MENTION_ENDS = []
for i in range(500):
    c = f'</m{i}>'
    MENTION_ENDS.append(c)
non_int_tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                              MENTION_START, COPY,
                              CLUSTER_NEW] +
                             MENTION_ENDS)
MENTION_ENDS_IDS = [non_int_tokenizer.encode(
    e,
    add_special_tokens=False)[0] for e in MENTION_ENDS]
END_TO_NUM = {e: i for i, e in enumerate(MENTION_ENDS)}
END_IDS_TO_NUM = {e: i for i, e in enumerate(MENTION_ENDS_IDS)}
NON_INT_SPECIAL_IDS = {
    'speaker_start': non_int_tokenizer.encode(
        SPEAKER_START,
        add_special_tokens=False)[0],
    'speaker_end':
        non_int_tokenizer.encode(
            SPEAKER_END, add_special_tokens=False)[0],
    'mention_start': non_int_tokenizer.encode(
        MENTION_START,
        add_special_tokens=False)[0],
    'cluster_ids': MENTION_ENDS_IDS,
    'cluster_ids_to_num': END_IDS_TO_NUM,
    'cluster_new': non_int_tokenizer.encode(
        CLUSTER_NEW,
        add_special_tokens=False)[0],
    'copy': non_int_tokenizer.encode(
        COPY, add_special_tokens=False)[0],
    'eos': non_int_tokenizer.eos_token_id
}