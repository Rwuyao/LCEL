from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
import chromadb
import logging
import sys

# 配置日志（输出到文件，便于排查闪退原因）
logging.basicConfig(
    filename='chroma_log.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
chroma_logger = logging.getLogger('chromadb')
chroma_logger.setLevel(logging.DEBUG)

# 异常捕获包装
try:
    embeddingUtils = DashScopeEmbeddings(
        dashscope_api_key="sk-52265888c7114eec8f039899b5759b6d",
        model="text-embedding-v4"
    )

    text_path = r"D:\demo.txt"  # 使用原始字符串避免转义问题
    loader = TextLoader(text_path, encoding='utf-8')
    docs = loader.load()
    print(f"加载的文档长度：{len(docs[0].page_content)} 字符")

    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"分割后的文档块数量：{len(all_splits)}")

    # 初始化 Chroma 客户端（移除不支持的 persist_interval 参数）
    client = chromadb.PersistentClient(
        path=r"D:\code\LCEL\chroma_db",
        settings=chromadb.Settings(
            anonymized_telemetry=False  # 仅保留支持的配置项 langchain_community chromadb dashscope
        ) 
    )
    collection = client.get_or_create_collection("my_collection")
    print("Chroma 集合创建成功")

    # 插入测试数据
    texts = ["我喜欢机器学习", "今天天气很好", "苹果是一家科技公司"]
    print("开始生成嵌入向量...")
    embeddings = embeddingUtils.embed_documents(texts)
    print(f"生成的嵌入向量数量：{len(embeddings)}")
    print(f"嵌入向量维度：{len(embeddings[0])}")  # 验证向量格式

    print("开始插入 Chroma 数据库...")
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=["1", "2", "3"]
    )
    print("数据插入成功")

    # 查询（修复 embed_query 用法：不需要传列表）
    query_text = "AI 和 深度学习"
    query_embedding = embeddingUtils.embed_query(query_text)  # 直接传字符串
    print(f"查询向量维度：{len(query_embedding)}")

    results = collection.query(
        query_embeddings=[query_embedding],  # query_embeddings 需要是列表
        n_results=2
    )
    print("查询结果：")
    print(results)
    print('程序执行完成')

except Exception as e:
    # 捕获所有异常并写入日志
    logging.error(f"程序异常：{str(e)}", exc_info=True)
    print(f"程序出错：{str(e)}")
    print("详细错误信息已写入 chroma_log.log 文件")
    sys.exit(1)