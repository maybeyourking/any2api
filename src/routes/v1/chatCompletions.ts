import { Router, type IRouter } from "express";
import { getCorsHeaders } from "./cors";
import { callAIResponse, callAIStream, type OpenAIMessage, type OpenAITool } from "./aiClient";

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
  const tools: OpenAITool[] | undefined = body.tools;
  const toolChoice: unknown = body.tool_choice;

  req.log.info({ model, stream: streamRequested, hasTools: !!tools }, "POST /v1/chat/completions");

  const id = generateId();
  const created = Math.floor(Date.now() / 1000);
  const options = { temperature, maxTokens, tools, toolChoice };

  if (streamRequested) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    try {
      await callAIStream(model, messages, options, res, id, created);
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

  try {
    const response = await callAIResponse(model, messages, options, id, created) as Record<string, unknown>;
    const choice = (response.choices as Array<Record<string, unknown>>)?.[0];
    const finishReason = choice?.finish_reason;
    const hasToolCalls = !!(choice?.message as Record<string, unknown>)?.tool_calls;
    req.log.info({ model, finishReason, hasToolCalls, msgCount: messages.length }, "Received AI response");
    res.json(response);
  } catch (err) {
    req.log.error({ err, model }, "AI call failed");
    res.status(500).json({
      error: { message: "AI model call failed", type: "api_error", code: "model_error" },
    });
  }
});

export default router;
