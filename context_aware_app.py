"""
Enhanced ML Model Service with Conversation Context Support
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Import LangChain components if available
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()

app = FastAPI(title="OBO Tutor ML Model Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[str] = ""
    chatbox_id: Optional[int] = None

# Initialize the model if Google API key is available
text_model = None
if LANGCHAIN_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
    try:
        text_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
            max_tokens=1500,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        print(f"Could not initialize Google AI model: {e}")

@app.get("/")
async def root():
    return {"message": "OBO Tutor ML Model Service v2.0 - Context-Aware!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-model", "context_support": True}

@app.post("/model/query/")
async def query_document(request: QueryRequest):
    """
    Process a question with conversation context and return an AI-generated response
    """
    try:
        question = request.question
        chat_history = request.chat_history or ""
        chatbox_id = request.chatbox_id
        
        if not question or len(question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # If we have LangChain and Google AI available, use it
        if text_model and LANGCHAIN_AVAILABLE:
            response = await generate_context_aware_response(question, chat_history)
        else:
            # Fallback to simple response with context awareness
            response = generate_simple_context_response(question, chat_history)
        
        return {
            "result": response,
            "relevant_images": [],
            "status": "success",
            "context_used": len(chat_history) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

async def generate_context_aware_response(question: str, chat_history: str) -> str:
    """Generate response using Google AI with conversation context"""
    try:
        # Create a context-aware prompt
        if chat_history.strip():
            prompt_template = """You are OboTutor, a friendly AI tutor. You're having a conversation with a student.

CONVERSATION CONTEXT:
{chat_history}

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Use the conversation context to understand what the student is referring to
- If they ask about "the project" or "it", refer to what was discussed earlier
- Maintain continuity with the previous conversation
- Be helpful and educational
- If you need clarification, ask follow-up questions

Respond naturally and conversationally:"""
        else:
            prompt_template = """You are OboTutor, a friendly AI tutor helping students learn.

QUESTION: {question}

INSTRUCTIONS:
- Be helpful and educational
- Explain concepts clearly
- Ask follow-up questions to better understand their needs
- Be encouraging and supportive

Respond naturally and conversationally:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | text_model | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        return response
        
    except Exception as e:
        print(f"Error with Google AI: {e}")
        return generate_simple_context_response(question, chat_history)

def generate_simple_context_response(question: str, chat_history: str) -> str:
    """Generate a simple response with basic context awareness"""
    
    # Check if question refers to something from context
    question_lower = question.lower()
    context_references = ["the project", "it", "this", "that", "overview", "what is it", "tell me more"]
    
    has_context_reference = any(ref in question_lower for ref in context_references)
    
    if has_context_reference and chat_history:
        # Try to extract project name or topic from chat history
        history_lower = chat_history.lower()
        
        if "roomba" in history_lower:
            if "overview" in question_lower or "project overview" in question_lower:
                return """Based on our conversation about Roomba, here's the project overview:

The Roomba project is about building an autonomous navigation robot using the OBO Car platform. It's designed to help students learn robotics programming through hands-on experience.

**Main Goals:**
• Create a robot that can navigate autonomously around a room
• Implement obstacle detection and avoidance
• Start with basic behaviors (turn when hitting obstacles)
• Progress to more advanced randomized navigation patterns

**Learning Objectives:**
• Understanding sensor integration
• Programming navigation algorithms  
• Hardware troubleshooting skills
• Autonomous robotics concepts

**Difficulty Level:** Beginner to intermediate - perfect for getting started with robotics programming!

Would you like me to explain any specific aspect of the Roomba project in more detail?"""
        
        # Check for other projects in history
        if "speech" in history_lower or "model" in history_lower:
            return """I can see from our conversation that we were discussing a speech-related project. Could you clarify which specific aspect you'd like me to explain? Are you asking about:

• The overall project goals?
• Technical implementation details?
• Learning outcomes?

This will help me give you the most relevant information!"""
    
    # Default response for questions without clear context
    if has_context_reference:
        return f"""I'd be happy to help with '{question}', but I need a bit more context. Could you let me know:

• What specific project or topic are you referring to?
• What particular aspect would you like me to explain?

This will help me give you the most accurate and helpful information!"""
    
    # Handle general questions
    return f"""I understand you're asking about: "{question}"

I'm here to help with your learning! To give you the best possible answer, could you provide a bit more detail about what you'd like to know?

I can help with:
• Explaining concepts and projects
• Breaking down complex topics
• Providing step-by-step guidance
• Answering specific technical questions

What would be most helpful for you?"""

@app.post("/model/upload/")
async def upload_document():
    """
    Document upload endpoint - placeholder for now
    """
    return {
        "message": "Document upload endpoint - to be implemented with context support",
        "status": "placeholder"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)