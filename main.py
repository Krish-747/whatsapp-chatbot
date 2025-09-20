# main.py

# Standard/third-party imports
from fastapi import FastAPI, Form, Depends
from decouple import config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Internal imports
from models import Conversation, SessionLocal
from utils import send_message, logger

app = FastAPI()

# Env vars
whatsapp_number = config("TO_NUMBER")

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# Memory (in-RAM, not persisted)
memory = ConversationBufferMemory(return_messages=True)

# Conversation chain
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# DB session dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# --- Helper: Rebuild memory from DB ---
def load_memory_from_db(db: Session, limit: int = 10):
    """Load last `limit` conversation turns into memory after restart."""
    history = (
        db.query(Conversation)
        .order_by(Conversation.id.desc())
        .limit(limit)
        .all()
    )
    for conv in reversed(history):  # oldest → newest
        memory.chat_memory.add_user_message(conv.message)
        memory.chat_memory.add_ai_message(conv.response)


@app.on_event("startup")
def restore_memory():
    """Rehydrate LangChain memory from DB on app startup."""
    db = SessionLocal()
    try:
        load_memory_from_db(db, limit=10)
        logger.info("ConversationBufferMemory restored from DB")
    finally:
        db.close()


# --- Webhook ---
@app.post("/message")
async def reply(Body: str = Form(""), db: Session = Depends(get_db)):
    # 1) Generate response with memory-enabled chain
    chat_response = await chain.apredict(input=Body)

    # 2) Store conversation in DB
    try:
        conversation = Conversation(
            sender=whatsapp_number,
            message=Body,
            response=chat_response
        )
        db.add(conversation)
        db.commit()
        logger.info(f"Conversation #{conversation.id} stored")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error storing conversation in DB: {e}")

    # 3) Reply via Twilio
    send_message(whatsapp_number, chat_response)
    return ""
