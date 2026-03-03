import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment variables")

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI()

# Allow all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# OpenRouter Client
# -------------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# -------------------------
# Request Model
# -------------------------
class PromptRequest(BaseModel):
    prompt: str
    tone: str
    mode: Optional[str] = "General"

# -------------------------
# Health Check Route
# -------------------------
@app.get("/")
def root():
    return {"status": "PromptSnap API Running"}

# -------------------------
# Enhance Endpoint
# -------------------------
@app.post("/enhance")
def enhance_prompt(data: PromptRequest):

    # Basic validation
    if len(data.prompt.strip()) < 5:
        raise HTTPException(status_code=400, detail="Prompt too short")

    logger.info(f"Enhancing prompt | Tone: {data.tone} | Mode: {data.mode}")

    # Mode-based behavior
    mode_instruction = ""

    if data.mode == "Code":
        mode_instruction = "Structure the prompt so it clearly instructs an AI to generate complete working code."
    elif data.mode == "Image":
        mode_instruction = "Optimize the prompt for image generation tools with vivid visual detail."
    elif data.mode == "Structured":
        mode_instruction = "Format the prompt in a clean, structured, bullet-point style."
    elif data.mode == "Startup":
        mode_instruction = "Transform the prompt into a compelling startup-style request."
    else:
        mode_instruction = "Enhance the prompt generally."
    
    system_prompt = f"""
        You are PromptSnap, an AI Prompt Enhancer.

        Your ONLY job is to rewrite the user's input into a clean,
        ready-to-copy, high-quality prompt.

        Tone: {data.tone}
        Mode: {data.mode}

        {mode_instruction}

        STRICT RULES:
        - Do NOT say "Here is your answer"
        - Do NOT add explanations
        - Do NOT add headings like "Improved Prompt:"
        - Do NOT speak to the user
        - Do NOT ask questions
        - Do NOT add markdown formatting
        - Output ONLY the final refined prompt text
        - The output must look like something the user can directly paste into ChatGPT or another AI

        Be concise but powerful.
        """

    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.prompt}
            ],
            temperature=0.7
        )

        enhanced = response.choices[0].message.content.strip()

        return {"enhanced_prompt": enhanced}

    except Exception as e:
        logger.error(f"AI Error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI service failed")