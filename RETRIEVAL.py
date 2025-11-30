# 导入 LangChain 库的不同模块，包括向量存储、输出解析器、提示模板、并行运行器和 OpenAI 的嵌入模型
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

embedding = DashScopeEmbeddings(
    dashscope_api_key="sk-xx",
    model="text-embedding-v4"
)

# 初始化 ChatOpenAI 模型，指定使用的模型为 'gpt-4o-mini'
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-xx",
    openai_api_base="https://api.deepseek.com"
)

# 使用 DocArrayInMemorySearch 创建一个内存中的向量存储
# 使用 OpenAIEmbeddings 为文本生成嵌入向量，文本为 "harrison worked at kensho" 和 "bears like to eat honey"
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=embedding,
)

# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()

# 创建一个聊天提示模板，用中文设置模板以便生成基于特定上下文和问题的完整输入
template = """根据以下上下文回答问题:
{context}

问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 初始化输出解析器，将模型输出转换为字符串
output_parser = StrOutputParser()

# 设置一个并行运行器，用于同时处理上下文检索和问题传递
# 使用RunnableParallel来准备预期的输入，通过使用检索到的文档条目以及原始用户问题，
# 利用文档搜索器 retriever 进行文档搜索，并使用 RunnablePassthrough 来传递用户的问题。
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# 构建一个处理链，包括上下文和问题的设置、提示生成、模型调用和输出解析
chain = setup_and_retrieval | prompt | model | output_parser

# 调用处理链，传入问题"where did harrison work?"（需翻译为中文），并基于给定的文本上下文生成答案
result = chain.invoke("harrison在哪里工作？")
print(result)

