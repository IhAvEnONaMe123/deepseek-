# Created by 蔡语轩
# 4/2/2025
# ollama要求配置，最好在macOS环境，理论其他系统也行 

import pyaudio
from vosk import Model, KaldiRecognizer
import requests
import json
import time
import pyttsx3

# ------------- 配置部分 -------------
MODEL_PATH = "models/vosk-model-small-cn-0.22"  # 这里换成你的离线模型目录
SAMPLE_RATE = 16000  # 模型通常是16k，如果你的模型支持其他采样率可自行调整
CHANNELS = 1         # 单声道
CHUNK = 4096         # 每次从麦克风读取的音频帧大小
SILENCE_TIMEOUT = 1.5  # 静音时长阈值，单位秒

engine = pyttsx3.init()

# ------------- 全局变量 -------------
user_input = ""
audio_text = ""
print_text1 = ""
print_text2 = ""
i = 0

# ------------- 函数部分 -------------
def init_vosk_model():
    """加载Vosk离线模型"""
    global i
    if i == 0:
        print(f"加载Vosk模型：{MODEL_PATH}")
    i = 1
    model = Model(MODEL_PATH)
    return model

def record_and_recognize(model):
    """
    <--由GPT完成-->
    从麦克风录音，并使用Vosk进行离线识别。
    当检测到超过SILENCE_TIMEOUT秒未识别到语音时，结束录音并返回识别结果。
    """

    # KaldiRecognizer(模型, 采样率)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)  # 如需词级别的结果可设为True

    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    stream.start_stream()

    print("正在录音，开始说话...（停止说话超过 %.1f 秒自动结束）" % SILENCE_TIMEOUT)

    last_spoke_time = time.time()
    recognized_text = ""

    while True:
        # 读取一批音频数据
        data = stream.read(CHUNK, exception_on_overflow=False)

        # 交给Recognizer处理
        if rec.AcceptWaveform(data):
            # 一旦Vosk认为这是一个完整语音片段，会返回一个JSON结果
            result_json = rec.Result()
            result = json.loads(result_json)
            # 这里的 "text" 是识别出来的完整句子
            text = result.get("text", "")
            if text.strip():
                recognized_text += (text + " ")
                # 更新最后说话时间
                last_spoke_time = time.time()
        else:
            # 如果还在说话中，会返回局部结果(Partial)
            partial_json = rec.PartialResult()
            partial = json.loads(partial_json).get("partial", "")
            if partial.strip():
                # 有新的局部语音，则说明用户还在说话
                last_spoke_time = time.time()

        # 检查是否超时
        current_time = time.time()
        if current_time - last_spoke_time > SILENCE_TIMEOUT:
            # 超过静音阈值，认为说话结束
            break

    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 最后一次FinalResult
    final_json = rec.FinalResult()
    final_result = json.loads(final_json).get("text", "")
    if final_result.strip():
        recognized_text += (final_result + " ")

    return recognized_text.strip()


def text_output():
    """核心和deepseek进行对接的位置\n较大的模型（14b）"""
    global print_text1, print_text2, audio_text
    # 向接口发起请求（/api/generate），于localhost:11434
    start_time = time.time()
    url = "http://127.0.0.1:11434/api/generate"
    headers = {"Content-Type":"application/json"}
    data = {
        "model": "deepseek-r1:14b",
        "prompt": user_input
    }
    # 流式读取
    response = requests.post(url=url, headers=headers, json=data, stream=True) 
    full_text = "" # 用于接收全部文本
    for line in response.iter_lines(decode_unicode=True):
        if line:
            # Ollama 返回的是一行一行的json，所以这边读的也是一行一行的json
            try:
                json_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"wrong, the output is {line}")
                continue
            if "response" in json_data:
                chunk = json_data["response"]
                full_text += chunk
                # 实现边生成边输出的效果 <--由GPT完成-->
                # print(chunk, end="")
            if "done" in json_data and json_data["done"]:
                break
    endtime = time.time()
    print("")
    print("="*40)
    # print("【最终生成内容】")
    try:
        words1 = full_text.split("<think>")
        words = words1[1].split("</think>")
        print_text1 = words[0]
        # print("【最终结果】",end='')
        # print(words[1])
        print_text2 = words[1]
        audio_text = words[1]
    except:
        pass
    # print(full_text)
    # print("【思考过程】")
    # print(words[0])
    
    # print_text1 = words[0]
    # # print("【最终结果】",end='')
    # # print(words[1])
    # print_text2 = words[1]
    # audio_text = words[1]
    # print(f"共用时{endtime-start_time}s")
    """
    while user_input != "/bye":
        xxx
        
    """

def text_output_1():
    """核心和deepseek进行对接的位置\n轻量化模型（1.5b）"""
    global print_text1, print_text2, audio_text
    # 向接口发起请求（/api/generate），于localhost:11434
    start_time = time.time()
    url = "http://127.0.0.1:11434/api/generate"
    headers = {"Content-Type":"application/json"}
    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": user_input
    }
    # 流式读取
    response = requests.post(url=url, headers=headers, json=data, stream=True) 
    full_text = "" # 用于接收全部文本
    for line in response.iter_lines(decode_unicode=True):
        if line:
            # Ollama 返回的是一行一行的json，所以这边读的也是一行一行的json
            try:
                json_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"wrong, the output is {line}")
                continue
            if "response" in json_data:
                chunk = json_data["response"]
                full_text += chunk
                # 实现边生成边输出的效果 <--由GPT完成-->
                # print(chunk, end="")
            if "done" in json_data and json_data["done"]:
                break
    endtime = time.time()
    print("")
    print("="*40)
    # print("【最终生成内容】")
    try:
        words1 = full_text.split("<think>")
        words = words1[1].split("</think>")
        print_text1 = words[0]
        # print("【最终结果】",end='')
        # print(words[1])
        print_text2 = words[1]
        audio_text = words[1]
    except:
        pass

def voice_output():
    """对用户语音进行识别\n返回值是最终识别的结果"""
    model = init_vosk_model()
    all_result_text = ""
    while True:
        cmd = input("\n回车开始录音，或输入q回车退出：")
        if cmd.strip().lower() == 'q':
            print("退出语音识别。")
            break
        
        # 调用录音+识别
        result_text = record_and_recognize(model)
        if result_text:
            print("识别结果：", result_text)
        else:
            print("未识别到有效语音。")
        
        all_result_text += result_text
    
    return all_result_text

def print_think():
    """文字输出"""
    print("【思考过程】", end="")
    print(print_text1)
    print("【最终结果】", end="")
    print(print_text2)

def print_think_1():
    """语音文字输出"""
    print(print_text2)

def audio_output():
    """语音输出"""
    engine.say(audio_text)
    engine.runAndWait()
    
# ------------- 主函数 -------------

if __name__ == "__main__":
    while True:
        a = input("****输入你想要的模型大小****\n1.比较大的模型（14b）\n2.比较轻量化的模型（1.5b）\n3.退出程序\n:")
        if a == "1":
            b = input("****输入你想用的模式****\n1.键盘输入文字输出\n2.键盘输入语音输出\n3.语音输入文字输出\n4.语音输入语音输出\n5.退出程序\n:")
            while True:
                if b == "1":
                    user_input = input(">>>")
                    if user_input == "/bye":
                        break
                    text_output()
                    print_think()
                elif b == "2":
                    user_input = input(">>>")
                    if user_input == "/bye":
                        break
                    text_output()
                    print_think_1()
                    audio_output()
                elif b == "3":
                    user_input = voice_output()
                    if user_input:
                        text_output()
                        print_think()
                    else:
                        break
                elif b == "4":
                    user_input = voice_output()
                    if user_input:
                        text_output()
                        print_think_1()
                        audio_output()
                    else:
                        break
                elif b == "5":
                    print("欢迎下次使用")
                    break
                else:
                    print("输入错误")
                    break
            break
        elif a == "2":
            b = input("****输入你想用的模式****\n1.键盘输入文字输出\n2.键盘输入语音输出\n3.语音输入文字输出\n4.语音输入语音输出\n5.退出程序\n:")
            while True:
                if b == "1":
                    user_input = input(">>>")
                    if user_input == "/bye":
                        break
                    text_output_1()
                    print_think()
                elif b == "2":
                    user_input = input(">>>")
                    if user_input == "/bye":
                        break
                    text_output_1()
                    print_think_1()
                    audio_output()
                elif b == "3":
                    user_input = voice_output()
                    if user_input:
                        text_output_1()
                        print_think()
                    else:
                        break
                elif b == "4":
                    user_input = voice_output()
                    if user_input:
                        text_output_1()
                        print_think_1()
                        audio_output()
                    else:
                        break
                elif b == "5":
                    print("欢迎下次使用")
                    break
                else:
                    print("输入错误")
                    break
            break
        elif a == "3":
            print("欢迎下次使用")
            break
        else:
            print("输入错误")
            continue