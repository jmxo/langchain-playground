import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser } from "@langchain/core/output_parsers"
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
// import { OpenAIEmbeddings } from "@langchain/openai";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { ChatOllama } from "@langchain/community/chat_models/ollama"
import { createRetrievalChain } from "langchain/chains/retrieval"

const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
)

const docs = await loader.load()
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs)

const embeddings = new OllamaEmbeddings({
  model: "llama2",
  maxConcurrency: 5
})

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
)

const chatModel = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama2"
})

const prompt = ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`)

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt
})

const retriever = vectorStore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever
})

const result = await retrievalChain.invoke({
  input: "what is LangSmith?"
})

console.log(result.answer)
