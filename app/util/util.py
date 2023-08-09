from g2p_en import G2p
from pyctcdecode import build_ctcdecoder
from .force_alignment import calculate_score
from .metric import Correct_Rate, align_for_force_alignment
from .help import remove_final_embedding_nucleus, text_to_phoneme
# from ...inference_for_gradio import inference_for_app
dict_vocab = {"u_2": 0, "uo_3": 1, "E_X_6a": 2, "ts_": 3, "O_X_2": 4, "ie_2": 5, "ie_1": 6, "o_3": 7, "dZ": 8, "e_4": 9, "E_X_3": 10, "ie_6a": 11, "o_4": 12, "M7_5a": 13, "t_h": 14, "uo_5a": 15, "7_X_5a": 16, "O_X_5b": 17, "kp": 18, "a_X_3": 19, "M_3": 20, "O_1": 21, "h": 22, "M_5b": 23, "a_5b": 24, "i_5b": 25, "O_5a": 26, "O_X_6b": 27, "u_5a": 28, "7_2": 29, "a_3": 30, "o_6a": 31, "a_X_2": 32, "i_5a": 33, "j": 34, "e_2": 35, "k": 36, "M_2": 37, "7_X_4": 38, "7_X_1": 39, "7_6a": 40, "a_X_6b": 41, "O_X_5a": 42, "M_5a": 43, "d": 44, "b": 45, "O_4": 46, "E_X_4": 47, "z": 48, "s": 49, "u_1": 50, "M7_3": 51, "e_6a": 52, "O_3": 53, "E_X_2": 54, "7_X_6b": 55, "uo_2": 56, "u_3": 57, "a_X_6a": 58, "o_5a": 59, "a_X_1": 60, "o_1": 61, "a_5a": 62, "o_5b": 63, "E_4": 64, "o_2": 65, "a_6a": 66, "i_1": 67, "O_X_4": 68, "e_5b": 69, "7_5b": 70, "E_5b": 71, "7_X_2": 72, "uo_4": 73, "ie_5b": 74, "M7_2": 75, "7_4": 76, "N": 77, "f": 78, "a_2": 79, "e_1": 80, "t": 81, "e_5a": 82, "tS": 83, "M7_4": 84, "E_X_5a": 85, "u_5b": 86, "S": 87, "m": 88, "w": 89, "r": 90, "a_X_4": 91, "uo_6b": 92, "a_X_5a": 93, "7_X_6a": 94, "ie_6b": 95, "E_6a": 96, "G": 97, "uo_5b": 98, "7_3": 99, "e_6b": 100, "M7_6b": 101, "i_6b": 102, "O_X_6a": 103, "7_1": 104, "v": 105, "M_6a": 106, "J": 107, "wp": 108, "M_4": 109, "ie_3": 110, "a_X_5b": 111, "M7_6a": 112, "7_6b": 113, "uo_1": 114, "u_6b": 115, "ie_5a": 116, "7_5a": 117, "E_1": 118, "E_X_1": 119, "o_6b": 120, "Nm": 121, "E_6b": 122, "u_6a": 123, "a_4": 124, "u_4": 125, "a_6b": 126, "E_2": 127, "7_X_5b": 128, "a_1": 129, "p": 130, "M7_5b": 131, "i_2": 132, "O_X_1": 133, "O_2": 134, "M_6b": 135, "E_X_6b": 136, "x": 137, "E_3": 138, "E_X_5b": 139, "M7_1": 140, "|": 141, "7_X_3": 142, "O_X_3": 143, "M_1": 144, "ie_4": 145, "O_6a": 146, "l": 147, "O_6b": 148, "uo_6a": 149, "i_6a": 150, "E_5a": 151, "i_3": 152, "i_4": 153, "O_5b": 154, "e_3": 155, "n": 156, " ": 157}
import sys
sys.path.append("....")


# phonemes_70 = [
#     'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
#     'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
#     'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
#     'EY2', 'F', 'G', 'HH',
#     'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
#     'M', 'N', 'NG', 'OW0', 'OW1',
#     'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
#     'UH0', 'UH1', 'UH2', 'UW',
#     'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
# ]


# ipa_mapping = {
#     'y': 'j', 'ng': 'ŋ', 'dh': 'ð', 'w': 'w', 'er': 'ɝ', 'r': 'ɹ', 'm': 'm', 'p': 'p',
#     'k': 'k', 'ah': 'ʌ', 'sh': 'ʃ', 't': 't', 'aw': 'aʊ', 'hh': 'h', 'ey': 'eɪ', 'oy': 'ɔɪ',
#     'zh': 'ʒ', 'n': 'n', 'th': 'θ', 'z': 'z', 'aa': 'ɑ', 'ao': 'aʊ', 'f': 'f', 'b': 'b', 'ih': 'ɪ',
#     'jh': 'dʒ', 's': 's', 'err': '', 'iy': 'i', 'uh': 'ʊ', 'ch': 'tʃ', 'g': 'g', 'ay': 'aɪ', 'l': 'l',
#     'ae': 'æ', 'd': 'd', 'v': 'v', 'uw': 'u', 'eh': 'ɛ', 'ow': 'oʊ'
# }

labels = ['u_2', 'uo_3', 'E_X_6a', 'ts_', 'O_X_2', 'ie_2', 'ie_1', 'o_3', 'dZ', 'e_4', 'E_X_3', 'ie_6a', 'o_4', 'M7_5a', 't_h', 'uo_5a', '7_X_5a', 'O_X_5b', 'kp', 'a_X_3', 'M_3', 'O_1', 'h', 'M_5b', 'a_5b', 'i_5b', 'O_5a', 'O_X_6b', 'u_5a', '7_2', 'a_3', 'o_6a', 'a_X_2', 'i_5a', 'j', 'e_2', 'k', 'M_2', '7_X_4', '7_X_1', '7_6a', 'a_X_6b', 'O_X_5a', 'M_5a', 'd', 'b', 'O_4', 'E_X_4', 'z', 's', 'u_1', 'M7_3', 'e_6a', 'O_3', 'E_X_2', '7_X_6b', 'uo_2', 'u_3', 'a_X_6a', 'o_5a', 'a_X_1', 'o_1', 'a_5a', 'o_5b', 'E_4', 'o_2', 'a_6a', 'i_1', 'O_X_4', 'e_5b', '7_5b', 'E_5b', '7_X_2', 'uo_4', 'ie_5b', 'M7_2', '7_4', 'N', 'f', 'a_2', 'e_1', 't', 'e_5a', 'tS', 'M7_4', 'E_X_5a', 'u_5b', 'S', 'm', 'w', 'r', 'a_X_4', 'uo_6b', 'a_X_5a', '7_X_6a', 'ie_6b', 'E_6a', 'G', 'uo_5b', '7_3', 'e_6b', 'M7_6b', 'i_6b', 'O_X_6a', '7_1', 'v', 'M_6a', 'J', 'wp', 'M_4', 'ie_3', 'a_X_5b', 'M7_6a', '7_6b', 'uo_1', 'u_6b', 'ie_5a', '7_5a', 'E_1', 'E_X_1', 'o_6b', 'Nm', 'E_6b', 'u_6a', 'a_4', 'u_4', 'a_6b', 'E_2', '7_X_5b', 'a_1', 'p', 'M7_5b', 'i_2', 'O_X_1', 'O_2', 'M_6b', 'E_X_6b', 'x', 'E_3', 'E_X_5b', 'M7_1', '|', '7_X_3', 'O_X_3', 'M_1', 'ie_4', 'O_6a', 'l', 'O_6b', 'uo_6a', 'i_6a', 'E_5a', 'i_3', 'i_4', 'O_5b', 'e_3', 'n', ' ']


# map_39 = {}
# for phoneme in phonemes_70:
#     phoneme_39 = phoneme.lower()
#     if phoneme_39[-1].isnumeric():
#         phoneme_39 = phoneme_39[:-1]
#     map_39[phoneme] = phoneme_39

tone = ['_1', '_2', '_3', '_4', '_5a', '_5b', '_6a', '_6b']

def text_to_phonemes(text):
    phonemes, _ = text_to_phoneme(text)
    # phonemes = remove_final_embedding_nucleus(phonemes)
    word_phoneme_in = []
    phonemes_result = []
    n_word = 0
    # print(phonemes)
    phonemes = phonemes.split()
    for phoneme in phonemes:
        # if map_39.get(phoneme, None) is not None:
        phonemes_result.append(phoneme)
        word_phoneme_in.append(n_word)
        if phoneme in tone:
            n_word += 1
    return ' '.join(phonemes_result), word_phoneme_in

print(text_to_phonemes("xin chào"))

def get_phoneme_ipa_form(text):
    phonemes, word_phoneme_in = text_to_phonemes(text.lower())
    phonemes = phonemes.split()
    result = ''
    for i in range(len(phonemes)):
        if i > 0 and word_phoneme_in[i] > word_phoneme_in[i - 1]:
            result += ' '
        # result += ipa_mapping[phonemes[i]]
        result += phonemes[i]
    return {'phonetics': result}

print(get_phoneme_ipa_form("xin chào"))


# def tokenizer_phonemes(phonemes):
#     text = phonemes.lower()
#     text = text.split(" ")
#     text_list = []
#     for idex in text:
#         text_list.append(dict_vocab[idex])
#     return text_list


decoder = build_ctcdecoder(
    labels = labels,
)

def decode(log_proba):
    return str(decoder.decode(log_proba)).strip()

def generate_mdd_for_app(log_proba, canonical, word_phoneme_in):
    emission = log_proba
    hypothesis = decode(emission).split()
    canonical = canonical.split()
    new_hypothesis = []
    for i in hypothesis:
        if i not in labels:
            continue
        else:
            canonical.a

    hypothesis_score = calculate_score(emission, hypothesis, dict_vocab)
    canonical_score = calculate_score(emission, canonical, dict_vocab)
    hypothesis_score, canonical_score = align_for_force_alignment(hypothesis_score, canonical_score)

    cnt, l, temp = Correct_Rate(canonical, hypothesis)
    correct_rate = 1 - cnt/l if l != 0 else 0

    result = [] # canonical, predict_phoneme, canonical_score, predict_score
    n = -1
    for i in range(len(canonical_score)):
        if canonical_score[i] != '<eps>':
            phoneme, score = canonical_score[i]
            n += 1
            if n == 0 or word_phoneme_in[n] > word_phoneme_in[n - 1]:
                result.append([])
            if isinstance(hypothesis_score[i], tuple):
                pred, predict_score = hypothesis_score[i]
            else:
                pred, predict_score = "<unk>", 0
            result[-1].append((
                phoneme,
                pred,
                score,
                predict_score
            ))
    
    return {
        'correct_rate': str(correct_rate),
        'phoneme_result': str(result)
    }


