import { Router, type IRouter } from "express";
import { getCorsHeaders } from "./cors";
import { callAI, callAIStream, type OpenAIMessage } from "./aiClient";

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
  const messages: OpenAIMessage[] = body.messages ?? [];
  const temperature: number | undefined = body.temperature;
  const maxTokens: number | undefined = body.max_tokens;
  const streamRequested: boolean = body.stream === true;

  req.log.info({ model, stream: streamRequested }, "POST /v1/chat/completions");

  const id = generateId();
  const created = Math.floor(Date.now() / 1000);

  if (streamRequested) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    try {
      await callAIStream(model, messages, { temperature, maxTokens }, res, id, created);
      req.log.info({ model }, "Stream completed");
    } catch (err) {
      req.log.error({ err, model }, "AI stream failed");
      if (!res.headersSent) {
        res.status(500).json({
          error: { message: "AI model call failed", type: "api_error", code: "model_error" },
        });
      } else {
        res.write(`data: ${JSON.stringify({ error: { message: "Stream error", type: "api_error" } })}\n\n`);
        res.end();
      }
    }
    return;
  }

  let content: string;
  try {
    content = await callAI(model, messages, { temperature, maxTokens });
    req.log.info({ contentLength: content.length }, "Received AI response");
  } catch (err) {
    req.log.error({ err, model }, "AI call failed");
    res.status(500).json({
      error: { message: "AI model call failed", type: "api_error", code: "model_error" },
    });
    return;
  }

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
});

export default router;
