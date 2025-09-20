# main.py

# Standard/third-party imports
from fastapi import FastAPI, Form, Depends, Response  # FastAPI core
from decouple import config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Internal imports
from models import Conversation, SessionLocal
from utils import send_message, logger

app = FastAPI()

# Env vars
# OPENAI_API_KEY is read automatically by langchain-openai via environment, so no direct openai.api_key assignment required.
whatsapp_number = config("TO_NUMBER")  # recipient number (e.g., user's WhatsApp E.164)

# Build a simple LangChain pipeline: prompt -> LLM -> string
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)  # choose any supported chat model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful WhatsApp assistant. Keep answers concise."),
    ("human", "{user_input}")
])
chain = prompt | llm | StrOutputParser()

# DB session dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# Webhook: Twilio posts application/x-www-form-urlencoded fields like Body, From, To
@app.post("/message")
async def reply(Body: str = Form(""), db: Session = Depends(get_db)):
    # 1) Generate response with LangChain (async-friendly)
    chat_response = await chain.ainvoke({"user_input": Body})

    # 2) Store the conversation
    try:
        conversation = Conversation(
            sender=whatsapp_number,
            message=Body,
            response=chat_response
        )
        db.add(conversation)
        db.commit()
        logger.info(f"Conversation #{conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error storing conversation in database: {e}")

    # 3) Reply via Twilio REST (out-of-band)
    send_message(whatsapp_number, chat_response)
    return ""