# 导入必要的库
import gc
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from chromadb.config import Settings

# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():

    embedding = DashScopeEmbeddings(
        dashscope_api_key="sk-xxx",
        model="text-embedding-v4"
    )

    # 初始化 ChatOpenAI 模型，指定使用的模型为 'gpt-4o-mini'
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key="sk-xxx",
        openai_api_base="https://api.deepseek.com"
    )

    # 使用 WebBaseLoader 从网页加载内容，并仅保留标题、标题头和文章内容
    text_path = "D:\西游记.txt"
    loader = TextLoader(text_path, encoding='utf-8')
    docs = loader.load()

    # 使用 RecursiveCharacterTextSplitter 将文档分割成块，每块1000字符，重叠200字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)


    # 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
    vectorstore = Chroma.from_documents(
                            documents=all_splits,
                            embedding=embedding,
                            persist_directory="./chroma_db"
                        )

    # 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    # 定义 RAG 链，将用户问题与检索到的文档结合并生成答案
    # 自定义提示词模板
    template = """你是一个专业的问答助手，将基于以下「参考文档片段」回答用户的问题。请严格遵循以下规则：

1. 回答必须完全基于提供的参考文档，不添加任何文档之外的无关信息，不编造事实；
2. 若参考文档中有多个相关片段，需整合所有关键信息，避免遗漏核心要点；
3. 若参考文档中没有与问题相关的内容，直接回复：「根据提供的参考信息，无法回答该问题。」，不随意猜测；
4. 回答结构清晰，优先用短句或分点（若信息较多），语言自然流畅，符合中文表达习惯；
5. 若问题涉及具体细节（如时间、数据、定义），需精准引用文档中的表述，确保准确性。

    参考文档片段：{context}

    用户问题： {question}

    请严格按照上述规则生成回答，确保准确性和可读性："""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # 使用 LCEL 构建 RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # 流式生成回答
    for chunk in rag_chain.stream("花果山在哪?"):
        print(chunk, end="", flush=True)
    print('第一个问题结束')

    # 流式生成回答
    for chunk in rag_chain.stream("猴王的名字是什么?"):
        print(chunk, end="", flush=True)
    print('第二个问题结束')

     # 流式生成回答
    for chunk in rag_chain.stream("?"):
        print(chunk, end="", flush=True)
    print('第三个问题结束')
if __name__ == "__main__":
    main()