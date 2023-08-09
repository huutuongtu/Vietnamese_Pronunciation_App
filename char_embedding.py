import json
from infer import remove_final_embedding_nucleus
d = 0
with open('vocab_embedding_nucleus.json') as f:
    d = json.load(f)


def text_to_tensor(canonical):
    str = remove_final_embedding_nucleus(canonical)
    # if "|" in str:
    #     text_list = []
    #     str = str.split("|")
    #     for syl in str:
    #         syl = syl.split(" ")
    #         # print(syl)
    #         for idex in syl:
    #             if len(idex)==0:
    #                 idex = ''
    #             else:
    #                 text_list.append(d[idex])
    #                 text_list.append(d[" "])
    #         text_list.pop()
    #         text_list.append(d["|"])
    #         text_list.append(d[" "])
    #     text_list.pop()
    #     text_list.pop()
    #     # t 7 _5a _5a| r 7_X _5b t _5b _5b| t_h i _5b k _5b _5b| v e _2 _2| k wp _1 e _1 _1|
    # else:    
    text = str
    # text = text.lower()
    text = text.split(" ")
    text_list = []
    for idex in text:
        text_list.append(d[idex])
        text_list.append(d[" "])
    text_list.pop()
    return text_list



key_list = list(d.keys())
val_list = list(d.values())

def tensor_to_text(ts):
    int_to_text = ts
    res = []
    for i in int_to_text:
        position = val_list.index(i)
        res.append(key_list[position])
    return res

def test_tensor_to_text(ts):
    int_to_text = ts
    res = ''
    for i in int_to_text:
        position = val_list.index(i)
        res = res + key_list[position]
    return res

# print((text_to_tensor("m o_6b t | d a_2 n | k O_2 | b E_X_6b k |")))
# print(test_tensor_to_text(text_to_tensor("d i _1 | s E _1 m _1 | d E_X _5a J _5a | k a _5a |")))

# print(text_to_tensor("a _5a"))
