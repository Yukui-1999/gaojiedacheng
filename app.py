import gradio as gr
import re
import subprocess
import os
import glob
import json
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_latest_file(directory, file_extension="*"):
    # 获取目录下所有文件
    files = glob.glob(os.path.join(directory, f"*.{file_extension}"))
    
    # 如果没有找到文件，返回None
    if not files:
        return None

    # 获取最新修改时间的文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file
def generate_travel_script(destination, days, budget, detail_level, randomness):
    # Sample script generation logic
    script = f"{destination}旅行，{days}天，{budget}元预算，{detail_level}描述"

    config_file_path = r'D:\documents\Code\Vscode\Python\xiaohongshupachong\MediaCrawler\config\base_config.py'
    new_keyword = script
    with open(config_file_path, 'r', encoding='utf-8') as file:
        config_content = file.read()
    new_config_content = re.sub(r'(KEYWORDS\s*=\s*")[^"]*(")', r'\1' + new_keyword + r'\2', config_content)
    with open(config_file_path, 'w', encoding='utf-8') as file:
        file.write(new_config_content)
    print("KEYWORDS 已成功更新为:", new_keyword)
    

    conda_env_name = 'gaojiedacheng'
    conda_env_python = r'C:\Users\Yukui\.conda\envs\gaojiedacheng\python.exe'
    if not os.path.isfile(conda_env_python):
        raise FileNotFoundError(f"Python interpreter not found: {conda_env_python}")
    main_py_path = r'D:\documents\Code\Vscode\Python\xiaohongshupachong\MediaCrawler\main.py'  # 替换为你的 main.py 的实际绝对路径
    if not os.path.isfile(main_py_path):
        raise FileNotFoundError(f"Script not found: {main_py_path}")
    command = [conda_env_python, main_py_path, '--platform', 'xhs', '--lt', 'cookie', '--type', 'search']
    result = subprocess.run(command, capture_output=True, text=True)
    print("标准输出:", result.stdout)
    print("标准错误:", result.stderr)
    print("返回码:", result.returncode)
    

    directory = r"D:\documents\Code\Vscode\Python\xiaohongshupachong\MediaCrawler\data\xhs"  # 替换为你的目录路径
    latest_file = get_latest_file(directory, "json")  # 替换为你的文件扩展名，例如"txt"、"py"等
    if latest_file:
        print("最新的文件是:", latest_file)
    else:
        print("目录中没有找到文件")

    # 加载JSON文件
    with open(latest_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取描述字段作为知识条目
    documents = [item['desc'] for item in data]

    # 定义要查询的问题
    query = "请生成一个关于{destination}{days}天{budget}元预算旅游的{detail_level}描述脚本"

    # 创建TF-IDF向量化器并将知识条目向量化
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # 将查询问题向量化
    query_vec = vectorizer.transform([query])

    # 计算余弦相似度
    similarities = cosine_similarity(query_vec, X).flatten()

    # 找到最相似的条目
    best_match_index = similarities.argmax()
    best_match = documents[best_match_index]

    # 定义提示词
    prompt = f"基于以下内容生成一个关于{destination}{days}天{budget}元预算旅游的{detail_level}描述脚本：\n{best_match}\n"

    api_key="sk-8F9n3GqEgKlV45Js7fE8Bf3285Bc47A6961035F272F3D256"
    api_base="https://api.aiwaves.cn/v1"
    client=OpenAI(api_key=api_key, base_url=api_base)

    stream = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
            "role": "system",
            "content": "你是一个旅行脚本生成器。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=randomness,
        stream=True,
    )
    result = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            result += content

    return result
    # for chunk in stream:
    #     if not chunk.choices:
    #         continue

    #     print(chunk.choices[0].delta.content, end="")
    # print()
    # return script

# Define interface components
destination_input = gr.Textbox(label="旅行目的地", value="重庆")
days_input = gr.Slider(label="天数", minimum=1, maximum=10, step=1, value=2)
budget_input = gr.Slider(label="预算（元）", minimum=100, maximum=10000, step=100, value=2000)
detail_level_input = gr.Radio(label="是否生成更详细的旅行攻略", choices=["简单", "详细"], value="简单")
randomness_input = gr.Slider(label="生成结果的随机度（越大越随机）", minimum=0, maximum=1, step=0.1, value=0)

# Define the Gradio interface
interface = gr.Interface(
    fn=generate_travel_script,
    inputs=[destination_input, days_input, budget_input, detail_level_input, randomness_input],
    outputs="text",
    title="旅行脚本生成器",
    description="输入旅行目的地、天数和预算，自动生成旅行脚本。",
    
)

# Launch the interface
interface.launch()