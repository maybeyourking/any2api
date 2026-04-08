import { Router, type IRouter } from "express";
import { getCorsHeaders } from "./cors";
import { callAI } from "./aiClient";

const router: IRouter = Router();

function generateId(): string {
  return "chatcmpl-" + Math.random().toString(36).slice(2, 11);
}

router.options("/v1/chat/completions", (req, res): void => {
  const headers = getCorsHeaders();
  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }
  res.sendStatus(204);
});

router.post("/v1/chat/completions", async (req, res): Promise<void> => {
  const corsHeaders = getCorsHeaders();
  for (const [key, value] of Object.entries(corsHeaders)) {
    res.setHeader(key, value);
  }

  const authHeader = req.headers["authorization"];
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.status(401).json({
      error: {
        message: "Missing or invalid Authorization header. Expected: Bearer <token>",
        type: "authentication_error",
        code: "invalid_api_key",
      },
    });
    return;
  }

  const body = req.body;
  if (!body || typeof body !== "object") {
    res.status(400).json({
      error: {
        message: "Invalid JSON body",
        type: "invalid_request_error",
        code: "invalid_request",
      },
    });
    return;
  }

  const model: string = body.model ?? "gpt-5";
  const messages: Array<{ role: "user" | "assistant" | "system"; content: string | unknown[] }> =
    body.messages ?? [];
  const temperature: number | undefined = body.temperature;
  const maxTokens: number | undefined = body.max_tokens;
  const streamRequested: boolean = body.stream === true;

  req.log.info({ model, stream: streamRequested }, "POST /v1/chat/completions");

  let content: string;
  try {
    content = await callAI(model, messages, { temperature, maxTokens });
    req.log.info({ contentLength: content.length }, "Received AI response");
  } catch (err) {
    req.log.error({ err, model }, "AI call failed");
    res.status(500).json({
      error: {
        message: "AI model call failed",
        type: "api_error",
        code: "model_error",
      },
    });
    return;
  }

  const id = generateId();
  const created = Math.floor(Date.now() / 1000);

  if (!streamRequested) {
    res.json({
      id,
      object: "chat.completion",
      created,
      model,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const roleChunk = JSON.stringify({
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  });
  res.write(`data: ${roleChunk}\n\n`);

  const words = content.split(/(\s+)/);
  const CHUNK_SIZE = 3;
  for (let i = 0; i < words.length; i += CHUNK_SIZE) {
    const chunk = words.slice(i, i + CHUNK_SIZE).join("");
    if (chunk.length === 0) continue;
    const contentChunk = JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }],
    });
    res.write(`data: ${contentChunk}\n\n`);
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
  }

  const finalChunk = JSON.stringify({
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  });
  res.write(`data: ${finalChunk}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
});

export default router;
