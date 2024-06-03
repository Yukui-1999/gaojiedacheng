import gradio as gr
import re
import subprocess
import os
import glob
import json
import time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import JSONLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from _config_ import _config_

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "sk-8F9n3GqEgKlV45Js7fE8Bf3285Bc47A6961035F272F3D256"
os.environ["OPENAI_API_BASE"] = "https://api.aiwaves.cn/v1"

def get_latest_file(directory, file_extension="*"):
    # 获取目录下所有文件
    files = glob.glob(os.path.join(directory, f"*.{file_extension}"))

    # 如果没有找到文件，返回None
    if not files:
        return None

    # 获取最新修改时间的文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def crawl_xhs(destination, days, budget, detail_level):
    # Sample script generation logic
    script = f"{destination}旅行，{days}天，{budget}元预算，{detail_level}描述"

    config_file_path = _config_.config_file_path
    new_keyword = script
    with open(config_file_path, 'r', encoding='utf-8') as file:
        config_content = file.read()
    new_config_content = re.sub(
        r'(KEYWORDS\s*=\s*")[^"]*(")', r'\1' + new_keyword + r'\2', config_content)
    with open(config_file_path, 'w', encoding='utf-8') as file:
        file.write(new_config_content)
    print("KEYWORDS 已成功更新为:", new_keyword)

    conda_env_name = _config_.conda_env_name
    conda_env_python = _config_.conda_env_python
    if not os.path.isfile(conda_env_python):
        raise FileNotFoundError(
            f"Python interpreter not found: {conda_env_python}")
    main_py_path = _config_.main_py_path
    if not os.path.isfile(main_py_path):
        raise FileNotFoundError(f"Script not found: {main_py_path}")
    command = [conda_env_python, main_py_path, '--platform',
               'xhs', '--lt', 'cookie', '--type', 'search']
    result = subprocess.run(command, capture_output=True, text=True)
    print("标准输出:", result.stdout)
    print("标准错误:", result.stderr)
    print("返回码:", result.returncode)

    directory = _config_.directory
    latest_file = get_latest_file(
        directory, "json")  # 替换为你的文件扩展名，例如"txt"、"py"等
    if latest_file:
        print("最新的文件是:", latest_file)
    else:
        print("目录中没有找到文件")
        
    return latest_file

# history
store = {}

class State:
    def __init__(self, destination="", days=0, budget=0, detail_level="简单"):
        self.destination = destination
        self.days = days
        self.budget = budget
        self.detail_level = detail_level
    def __eq__(self, other):
        # 首先检查 'other' 是否是 TravelPlan 类的一个实例
        if not isinstance(other, State):
            # 如果不是，返回 False
            return NotImplemented

        # 比较所有属性
        return (self.destination == other.destination and
                self.days == other.days and
                self.budget == other.budget and
                self.detail_level == other.detail_level)
        
cur_state = State()
rag_chain = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_rag_chain(destination, days, budget, detail_level, randomness):
    llm = ChatOpenAI(
        model="gpt-4-0125-preview",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=randomness
    )
    
    # retriever
    loader = JSONLoader(
        # file_path=crawl_xhs(destination, days, budget, detail_level),
        file_path=_config_.dev_json_file,
        jq_schema='.[].desc',
        # content_key='uri'
        text_content=False)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    
    # Contextualize
    contextualize_q_system_prompt = (
        "根据对话历史和用户最新提出的问题，"
        "这个问题可能涉及对话历史中的上下文。"
        "请重新构造一个独立的、不依赖于对话历史即可理解的问题。"
        "如果需要，请重新表述问题；如果不需要，就原样返回。"
        "不要回答问题。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "你是一个旅游行程规划助手。"
        "使用以下检索到的上下文(context)片段来回答这个问题。"
        "如果你不知道答案，就说你不知道。"
        "如果你觉得问题中的要求不合理，请直接指出。"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain
    
def generate_travel_script(destination, days, budget, detail_level, randomness, other):
    global cur_state
    global rag_chain
    tmp_state = State(destination, days, budget, detail_level)
    if tmp_state == cur_state and other != '':
        query = other
    else:
        rag_chain = generate_rag_chain(destination, days, budget, detail_level, randomness)
        
        # 定义要查询的问题  
        # query = "请生成一个关于重庆2天2000元预算旅游的简单描述脚本"
        query = f"请生成一个关于{destination}{days}天{budget}元预算旅游的{detail_level}描述脚本，如果可以的话请同时给出具体开支、游玩时长、移动距离。{other}"
        cur_state.__dict__ = tmp_state.__dict__.copy()
    
    result = ''
    for s in rag_chain.stream(
        {"input": query},
        config={"configurable": {"session_id": "travelplan_lzf"}},
    ):
        if 'answer' in s:   
            result=result+s['answer']
            yield result


# Define interface components
destination_input = gr.Textbox(label="旅行目的地", value="重庆")
days_input = gr.Slider(label="天数", minimum=1, maximum=14, step=1, value=2)
budget_input = gr.Slider(label="预算（元）", minimum=100,
                         maximum=10000, step=100, value=2000)
detail_level_input = gr.Radio(label="是否生成更详细的旅行攻略", choices=[
                              "简单", "详细"], value="简单")
randomness_input = gr.Slider(
    label="生成结果的随机度（越大越随机）", minimum=0, maximum=1, step=0.1, value=0)
other_input = gr.Textbox(label="备注", value="")

# Define the Gradio interface
interface = gr.Interface(
    fn=generate_travel_script,
    # fn=empty,
    inputs=[destination_input, days_input, budget_input,
            detail_level_input, randomness_input,other_input],
    outputs="text",
    title="旅行脚本生成器",
    description="输入旅行目的地、天数和预算，自动生成旅行脚本。"
)

# Launch the interface
interface.launch()
