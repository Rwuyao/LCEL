import chromadb
import logging
import sys

# 超详细日志配置（捕获 chromadb 底层 C 扩展日志）
logging.basicConfig(
    filename='chroma_minimal_log.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制覆盖已有日志配置
)
# 捕获 chromadb 所有子模块日志（包括底层 C 扩展）
for name in logging.root.manager.loggerDict:
    if 'chroma' in name.lower() or 'llama' in name.lower():
        logging.getLogger(name).setLevel(logging.DEBUG)

try:
    print("=== 初始化 Chroma 内存客户端 ===")
    # 使用内存模式（跳过磁盘 IO，排除权限/路径问题）
    client = chromadb.EphemeralClient(
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("test_collection")
    print("✅ 集合创建成功")

    print("\n=== 插入测试数据 ===")
    # 手动构造简单嵌入向量（避免依赖 DashScope，排除嵌入生成问题）
    texts = ["测试文本1", "测试文本2"]
    embeddings = [[0.1]*768, [0.2]*768]  # 模拟 768 维向量（text-embedding-v4 维度）
    collection.add(
        ids=["1", "2"],
        documents=texts,
        embeddings=embeddings
    )
    print("✅ 数据插入成功")

    print("\n=== 执行查询 ===")
    query_embedding = [0.15]*768
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    print("✅ 查询成功，结果：")
    print(results)
    print("\n🎉 最小化代码运行正常！原闪退问题不在 chromadb 核心功能")

except Exception as e:
    logging.error(f"❌ 程序异常：{str(e)}", exc_info=True)
    print(f"❌ 出错：{str(e)}")
    print("详细日志已保存到 chroma_minimal_log.log")
    sys.exit(1)