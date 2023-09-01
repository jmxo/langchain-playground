import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser } from "@langchain/core/output_parsers"

const chatModel = new ChatOpenAI({});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"]
])

const outputParser = new StringOutputParser()

const chain = prompt.pipe(chatModel).pipe(outputParser);

const result = await chain.invoke({
  input: "what is LangSmith?"
})

console.log(result)
