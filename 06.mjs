import { ChatOpenAI } from "@langchain/openai"
import { StringOutputParser } from "@langchain/core/output_parsers"
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { ChatOllama } from "@langchain/community/chat_models/ollama"
import { createRetrievalChain } from "langchain/chains/retrieval"
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever"
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts"
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createRetrieverTool } from "langchain/tools/retriever"
import { TavilySearchResults } from "@langchain/community/tools/tavily_search"
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";


const searchTool = new TavilySearchResults();

const tools = [searchTool]

const agentPrompt = await pull < ChatPromptTemplate > (
    "hwchase17/openai-functions-agent"
)

const agentModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    tools,
    prompt: agentPrompt
})

const agent = await createOpenAIFunctionsAgent({
    llm: agentModel,
    tools,
    prompt: agentPrompt,
});

const agentExecutor = new AgentExecutor({
    agent,
    tools,
    verbose: true
})


const result = await agentExecutor.invoke({
    input: "what is the weather in SF?"
})

console.log(result.output)
