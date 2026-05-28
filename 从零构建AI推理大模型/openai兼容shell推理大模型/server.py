import json
import os
import time
import uuid
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from config import MODEL_CONFIG, SERVER_CONFIG, PATHS, SPECIAL_TOKENS
from tokenizer import BPETokenizer
from model import GPTModel
from dataset import format_input


app = FastAPI(title="Custom LLM API", description="OpenAI-compatible API for custom trained LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
device = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = SERVER_CONFIG["model_name"]
    messages: List[ChatMessage]
    temperature: Optional[float] = SERVER_CONFIG["temperature"]
    top_p: Optional[float] = SERVER_CONFIG["top_p"]
    max_tokens: Optional[int] = SERVER_CONFIG["max_new_tokens"]
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


def load_model_and_tokenizer():
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    tokenizer = BPETokenizer()
    tokenizer.load()

    if os.path.exists(PATHS["best_model"]):
        checkpoint = torch.load(PATHS["best_model"], map_location=device)
        config = checkpoint.get("model_config", MODEL_CONFIG)
        model = GPTModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from {PATHS['best_model']}")
    else:
        print(f"WARNING: No trained model found at {PATHS['best_model']}")
        MODEL_CONFIG["vocab_size"] = len(tokenizer)
        model = GPTModel(MODEL_CONFIG)

    model.to(device)
    model.eval()
    print("Model loaded successfully!")


def run_inference(prompt: str, max_tokens: int = 128, temperature: float = 0.1,
                  top_p: float = 0.9) -> str:
    input_text = format_input(prompt)
    input_ids = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long).to(device)

    output_ids = model.generate(
        input_ids, tokenizer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
    )

    decoded = tokenizer.decode(output_ids)

    if "<eos>" in decoded:
        parts = decoded.split("<eos>")
        if len(parts) > 1:
            result = parts[1].strip()
        else:
            result = decoded.replace("<bos>", "").replace("<eos>", "").strip()
    else:
        result = decoded.replace("<bos>", "").strip()

    if "<eos>" in result:
        result = result.split("<eos>")[0].strip()

    return result


def execute_tool_call(tool_call_str: str) -> str:
    try:
        tool_call = json.loads(tool_call_str)
        func_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        if func_name == "get_weather":
            city = arguments.get("city", "Unknown")
            return json.dumps({
                "city": city,
                "temperature": "22°C",
                "condition": "Partly Cloudy",
                "humidity": "65%",
                "wind": "12 km/h",
            }, indent=2)

        elif func_name == "list_directory":
            path = arguments.get("path", "/")
            return json.dumps({
                "path": path,
                "contents": [
                    {"name": "documents", "type": "directory"},
                    {"name": "downloads", "type": "directory"},
                    {"name": "readme.txt", "type": "file", "size": "1.2KB"},
                    {"name": "config.json", "type": "file", "size": "0.5KB"},
                    {"name": "data.csv", "type": "file", "size": "3.8KB"},
                ],
            }, indent=2)

        else:
            return json.dumps({"error": f"Unknown function: {func_name}"})

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid function call format"})


@app.on_event("startup")
async def startup_event():
    load_model_and_tokenizer()


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend.html"))


@app.get("/v1/models")
async def list_models():
    return ModelsResponse(
        data=[
            ModelInfo(
                id=SERVER_CONFIG["model_name"],
                created=int(time.time()),
                owned_by="custom",
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_message = ""
    for msg in request.messages:
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    try:
        raw_output = run_inference(
            user_message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        try:
            tool_call = json.loads(raw_output)
            func_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

            tool_result = execute_tool_call(raw_output)

            response_content = f"Function called: {func_name}\nArguments: {json.dumps(arguments)}\n\nResult:\n{tool_result}"

            message = ChatMessage(
                role="assistant",
                content=response_content,
            )

        except (json.JSONDecodeError, KeyError):
            message = ChatMessage(
                role="assistant",
                content=raw_output if raw_output else "I couldn't understand that request.",
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(tokenizer.encode(user_message)),
                completion_tokens=len(tokenizer.encode(message.content)),
                total_tokens=len(tokenizer.encode(user_message)) + len(tokenizer.encode(message.content)),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: dict):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 128)
    temperature = request.get("temperature", 0.1)
    top_p = request.get("top_p", 0.9)

    try:
        output = run_inference(prompt, max_tokens, temperature, top_p)
        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": SERVER_CONFIG["model_name"],
            "choices": [
                {
                    "text": output,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(output)),
                "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(output)),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
    )


if __name__ == "__main__":
    start_server()
