import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import {
    RunnableSequence,
    RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { MemoryVectorStore } from "langchain/vectorstores/memory"

const model = new ChatOpenAI({});

const CONDENSE_QUESTION_PROMPT = PromptTemplate.fromTemplate(
    `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`
);

const ANSWER_PROMPT = PromptTemplate.fromTemplate(
    `Answer the question based only on the following context:
{context}

Question: {question}
`
);

const formatChatHistory = chatHistory => chatHistory.map(
    dialogueTurn => `Human: ${dialogueTurn[0]}\nAssistant: ${dialogueTurn[1]}`
).join("\n");

const vectorStore = await MemoryVectorStore.fromTexts(
    [
        "mitochondria is the powerhouse of the cell",
        "mitochondria is made of lipids",
    ],
    [{ id: 1 }, { id: 2 }],
    new OpenAIEmbeddings()
);
const retriever = vectorStore.asRetriever();

const standaloneQuestionChain = RunnableSequence.from([
    {
        question: input => input.question,
        chat_history: input => formatChatHistory(input.chat_history),
    },
    CONDENSE_QUESTION_PROMPT,
    model,
    new StringOutputParser(),
]);

const answerChain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
    },
    ANSWER_PROMPT,
    model,
]);

const conversationalRetrievalQAChain =
    standaloneQuestionChain.pipe(answerChain);

const result1 = await conversationalRetrievalQAChain.invoke({
    question: "What is the powerhouse of the cell?",
    chat_history: [],
});
console.log(result1);

const result2 = await conversationalRetrievalQAChain.invoke({
    question: "What are they made out of?",
    chat_history: [
        [
            "What is the powerhouse of the cell?",
            "The powerhouse of the cell is the mitochondria.",
        ],
    ],
});

console.log(result2);
