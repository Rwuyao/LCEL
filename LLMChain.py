from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 初始化 ChatOpenAI 模型，指定使用的模型为 'gpt-4o-mini'
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-9f4b5c77f02d4b22a23c4a4aa4a10054",
    openai_api_base="https://api.deepseek.com"
)

# 创建一个聊天提示模板，设置模板内容为"讲个关于 {topic} 的笑话吧"
prompt = ChatPromptTemplate.from_template("讲个关于 {topic} 的笑话吧")

# 初始化输出解析器，用于将模型的输出转换为字符串
output_parser = StrOutputParser()

# 构建一个处理链，先通过提示模板生成完整的输入，然后通过模型处理，最后解析输出
chain = prompt | model | output_parser

# 调用处理链，传入主题为"程序员"，生成关于程序员的笑话
result = chain.invoke({"topic": "漩涡忍传"})
print(result)