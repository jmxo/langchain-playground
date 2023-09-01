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

const retriever = vectorStore.asRetriever();

const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: "llama2"
})

const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
        "user",
        "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"
    ]
])

const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: historyAwarePrompt
})

const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        "Answer the user's questions based on the below context:\n\n{context}"
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"]
])

const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt
})

const conversationalRetrievalChain = await createRetrievalChain({
    retriever: historyAwareRetrieverChain,
    combineDocsChain: historyAwareCombineDocsChain
})

const result = await conversationalRetrievalChain.invoke({
    chat_history: [
        new HumanMessage("Can LangSmith help test my LLM applications?"),
        new AIMessage("Yes!")
    ],
    input: "tell me how"
})

console.log(result.answer)
