import json
# from infer import remove_final_embedding_nucleus
d = 0
with open('vocab_embedding_nucleus.json') as f:
    d = json.load(f)


f = open("D:/DATN/Interspeech2023/Vietnamese_fixkaldifilterbank/Embedding_nucleus/app/util/vi_SG_lexicon.dict", "r", encoding='utf8')
data = f.readlines()
init = []
PHONEME = []
WORD = []
for i in range(len(data)):
    word = data[i].split("|")[0]
    WORD.append(word)
    phoneme = (data[i].split("|")[1].split("\n")[0])
    init.append(phoneme.split(" ")[0])
    PHONEME.append(phoneme)

def text_to_phoneme(canonical):
    res = ''
    seq = canonical.split(" ")
    for text in seq:
        res = res + PHONEME[WORD.index(text)] + " "
    return res.strip(), canonical




NUCLEAR = ['a', 'E', 'e', 'i', 'O', 'o', '7', 'u', 'M', 'a_X', '7_X', 'E_X', 'O_X', 'ie', 'uo', 'M7']
tone = ['_1', '_2', '_3', '_4', '_5a', '_5b', '_6a', '_6b']
EMBEDDING_NUCLEAR = []
RAW = []

for nucl in NUCLEAR:
    for tonal in tone:
        RAW.append(nucl + " " + tonal)
        EMBEDDING_NUCLEAR.append(nucl + tonal)

def reconstruct_remove_final_add_nucleous(phoneme):
    res = ''
    phoneme = phoneme.split("|")
    for word in phoneme:
        t = ''
        word = word.split(" ")
        for char in word:
            if char in tone:
                t = char + " "
            else:
                res = res + char + " "
        res = res + t
    return res.strip().replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")

def reconstruct_remove_final_embedding_nucleus(phoneme):
    remove_final_add_nucleous = ''
    phoneme = phoneme.split("|")
    for word in phoneme:
        word_split = word.split(" ")
        for char in word_split:
            if char in EMBEDDING_NUCLEAR:
                # print(char)
                id = EMBEDDING_NUCLEAR.index(char)
                word = word.replace(EMBEDDING_NUCLEAR[id], RAW[id])
        remove_final_add_nucleous = remove_final_add_nucleous + word + "|"
    return reconstruct_remove_final_add_nucleous(remove_final_add_nucleous)


def remove_final_embedding_nucleus(canonical):
    res = ''
    tone = ["_1", "_2", "_3", "_4", "_5a", "_5b", "_6a", "_6b"]
    phoneme = canonical.split(" ")
    arr = []
    j = 0
    cnt = 0
    for i in phoneme:
        if i in tone:
            arr.append("")
    for i in phoneme:
        if i not in tone:
            arr[j] = arr[j] + " " + i
        else:
            arr[j] = arr[j] + " " + i
            arr[j] = arr[j].strip()
            j = j + 1
    for word in arr:
        tmp = ''
        word = word.split(" ")
        for char in word:
            if char not in NUCLEAR and char not in tone:
                tmp = tmp + char + " "
            elif char not in tone:
                tmp = tmp + char + word[-1] + " "
        arr[cnt] = tmp + "|" + " "
        cnt = cnt + 1
    for word in arr:
        res = res + word
    return res.strip()




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
