import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"

const chatModel = new ChatOpenAI({});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"]
])

const chain = prompt.pipe(chatModel);

const result = await chain.invoke({
  input: "what is LangSmith?"
})

console.log(result)
