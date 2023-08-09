import gradio as gr
from inference_for_gradio import inference
from metric import Align, Correct_Rate
vocab_back = {"j":"i", "b": "b", "d": "đ", "s": "x", "S": "s", "g": "g,gh", "x": "kh", "l": "l", "v": "v", "t_h": "th", "z": "d", "dZ": "gi", "r": "r", "f": "ph", "tS": "tr", "ts_": "ch", "h": "h", "k": "c,k,q,ch", "t": "t", "p": "p", "n": "n", "kp": "c", "m": "m", "J": "nh", "N": "ng,ngh", "Nm": "ng", "wp": "o,u", "w": "o,u", "a": "a", "E": "e", "e": "ê", "i": "i,y", "O": "o,oo", "o": "ô", "7": "ơ", "u": "u", "M": "ư", "a_X": "ă,a", "7_X": "â", "E_X": "a", "O_X": "o", "ie": "ia,iê,yê,ya", "uo": "ua,uô", "M7": "ưa,ươ", "_1": "ngang", "_2": "huyền", "_3": "hỏi", "_4": "ngã", "_5b": "sắc", "_5a": "sắc", "_6a": "nặng", "_6b": "nặng"}

posible_phoneme = list(vocab_back.keys())

tone = ["_1","_2","_3","_4","_5a","_5b","_6a","_6b"]
def Align_for_app(groundtruth: str, hypothesis: str, text_raw):
    text = text_raw
    text = text.split(" ")
    res = {}
    tone_gt = []
    gt_split_by_word = [[]]
    groundtruth = groundtruth.split(" ")
    new_hypothesis = []
    for phoneme in groundtruth:
        if phoneme not in tone:
            gt_split_by_word[-1].append(phoneme)
        if phoneme in tone:
            tone_gt.append(phoneme)
            gt_split_by_word[-1].append(phoneme)
            gt_split_by_word.append([])
    gt_split_by_word.pop()
    X,Y = Align(groundtruth, hypothesis.split())
    new_gt = []
    op = []
    for i in range(len(X)):
        if X[i] == "<eps>" and Y[i]!="<eps>":
                continue
        elif X[i]!="<eps>" and Y[i]=="<eps>":
            new_hypothesis.append(Y[i])
            op.append("D")
        elif (X[i]!=Y[i]) and Y[i]!="<eps>" and X[i]!="<eps>":
            new_hypothesis.append(Y[i])
            op.append("S")
        else:
            new_hypothesis.append(Y[i])
            op.append("C")

    res = ''
    cnt = 0
    new_gt = gt_split_by_word

    for i in range(len(new_gt)):
        for j in range(len(new_gt[i])):
            if op[cnt] == "D":
                res = res + "Đọc thiếu âm /" + vocab_back[new_gt[i][j]] + "/ tại từ thứ " + str(i+1) + ":" + text[i] + "\n"
            elif op[cnt] == "S":
                if new_hypothesis[cnt] not in posible_phoneme:
                    res = res + "Đọc sai âm /" + vocab_back[new_gt[i][j]] + "/ tại từ thứ " + str(i+1) + ":" + text[i] + "\n"
                else:
                    res = res + "Đọc sai âm /" + vocab_back[new_gt[i][j]] + "/ thành âm /" + vocab_back[new_hypothesis[cnt]] + "/ tại từ thứ " + str(i+1) + ":" +  text[i] + "\n"
            cnt = cnt + 1
    return res

def transcribe(audio, text):
    Text_Raw = text
    flag = True
    k = ''
    hypothesis, groundtruth, raw_sequence = inference(audio, text)
    print(hypothesis)
    print(groundtruth)
    ref = groundtruth
    hyp = hypothesis
    X, Y = groundtruth, hypothesis
    res = ''
    res = res + "Câu gốc: " + raw_sequence + "\n"
    res = res + "Âm vị câu gốc:      " + X + "\n"

    groundtruth, hypothesis = Align(X.split(" "), Y.split(" "))
    number_wrong, total_phoneme, _ = Correct_Rate(X.split(" "), Y.split(" "))
    CR = (1 - number_wrong/total_phoneme)*100
    for idex in range(len(hypothesis)):
        if groundtruth[idex] == "<eps>" and hypothesis[idex]!="<eps>" :
            k = k
        elif groundtruth[idex]!="<eps>" and hypothesis[idex]=="<eps>":
            k = k + hypothesis[idex] + " "
        elif (groundtruth[idex]!=hypothesis[idex]) and hypothesis[idex]!="<eps>" and groundtruth[idex]!="<eps>":
            k = k + hypothesis[idex] + " "
        else:
            k = k + hypothesis[idex] + " "
    k = k.strip().replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
    res = res + "Âm vị nhận dạng: " + k + "\n"
    res = res + "Bạn nói giống người bản ngữ " + str(CR) + "% \n" 
    res = res + "Phân tích lỗi:" + "\n"
    res = res + Align_for_app(ref,hyp,Text_Raw)

    """
    for idex in range(len(hypothesis)):
        # new_hypothesis
        # if groundtruth[idex] == "<eps>" and hypothesis[idex]!="<eps>" :
        # #     res = res + "I" + "  "*3
        #     # new_hypothesis[idex] = ""
        #     hypothesis[idex] = hypothesis[idex].strip()
        if groundtruth[idex]!="<eps>" and hypothesis[idex]=="<eps>":
            res = res + "Đọc thiếu " + str(groundtruth[idex]) +  " âm vị thứ: " + str(idex) + "\n"
            flag = False
            # k = k + hypothesis[idex] + " "
        elif (groundtruth[idex]!=hypothesis[idex]) and hypothesis[idex]!="<eps>" and groundtruth[idex]!="<eps>":
            res = res + "Đọc sai " + str(groundtruth[idex]) + " thành " + str(hypothesis[idex])+ " âm vị thứ: " + str(idex) + "\n"
            flag = False
        """
    # if flag:
    #     res = res + "Không có lỗi" + "\n"
    #     res = res + "Bạn đọc giống người bản ngữ 100%"
    # else:
    #     cnt, length, _ = Correct_Rate(X.split(" "), Y.split(" ")) 
    #     res = res + "Bạn đọc giống người bản ngữ " + str(100*(1-cnt/length)) + "%"

    return res

# print(transcribe('demo_audio/toidenaithanhaitoi.wav', 'tôi đến thăm anh hai tôi'))


# gr.Interface(
#     fn=transcribe, 
#     inputs=[gr.Audio(source="microphone", type = 'filepath'), 'text'], 
#     # inputs = [gr.Audio(type = 'filepath'), 'text'],
#     outputs="text").launch()


# gr.Interface(
#     fn=transcribe, 
#     inputs=[gr.Audio(type='filepath'), 'text'], 
#     # inputs = [gr.Audio(type = 'filepath'), 'text'],
#     outputs="text").launch()


gr.Interface(
    fn=transcribe, 
    inputs=[gr.Audio(source="microphone", type='filepath'), 'text'], 
    # inputs = [gr.Audio(type = 'filepath'), 'text'],
    outputs="text").launch(share = False, ssl_verify=True)
