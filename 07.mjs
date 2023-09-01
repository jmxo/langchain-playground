import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import {
    RunnableSequence,
    RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { ChatOllama } from "@langchain/community/chat_models/ollama"
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { MemoryVectorStore } from "langchain/vectorstores/memory"

const model = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: "llama2"
})

const embeddings = new OllamaEmbeddings({
    model: "llama2",
    maxConcurrency: 5
})

const vectorStore = await MemoryVectorStore.fromTexts(
    ["mitochondria is the powerhouse of the cell"],
    [{ id: 1 }],
    embeddings
);

const retriever = vectorStore.asRetriever();

const prompt = PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

const chain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
]);

const result = await chain.invoke("What is the powerhouse of the cell?");

console.log(result);
