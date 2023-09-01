import { ChatOpenAI } from "@langchain/openai"

const chatModel = new ChatOpenAI({});
const result = await chatModel.invoke("what is LangSmith?")
console.log(result)
