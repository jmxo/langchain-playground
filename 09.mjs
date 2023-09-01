import 'dotenv/config'
import "cheerio";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { PromptTemplate, ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";
import {
    RunnableSequence,
    RunnablePassthrough,
} from "@langchain/core/runnables";


const loader = new CheerioWebBaseLoader(
    "https://en.wikipedia.org/wiki/Large_language_model",
    {
        selector: "p",
    }
);

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const splits = await textSplitter.splitDocuments(docs);

const vectorStore = await MemoryVectorStore.fromDocuments(
    splits,
    new OpenAIEmbeddings({ OPENAI_API_KEY: process.env.OPENAI_API_KEY })
);

const retriever = vectorStore.asRetriever();

const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0 });

const contextualizeQSystemPrompt = `Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.`;

const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
]);

const contextualizeQChain = contextualizeQPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());

const qaSystemPrompt = `You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}`;

const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
]);

const contextualizedQuestion = input => {
    if ("chat_history" in input) {
        return contextualizeQChain;
    }
    return input.question;
};

const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
        context: input => {
            if ("chat_history" in input) {
                const chain = contextualizedQuestion(input);
                return chain.pipe(retriever).pipe(formatDocumentsAsString);
            }
            return "";
        },
    }),
    qaPrompt,
    llm,
]);

let chat_history = [];

const question = "What is task decomposition?";
const aiMsg = await ragChain.invoke({ question, chat_history });
console.log(aiMsg);
chat_history = chat_history.concat(aiMsg);

const secondQuestion = "What are common ways of doing it?";
await ragChain.invoke({ question: secondQuestion, chat_history });
