import { Router, type IRouter } from "express";
import { getCorsHeaders } from "./cors";
import { callAI, callAIStreamAnthropic, type OpenAIMessage } from "./aiClient";

const router: IRouter = Router();

function generateMsgId(): string {
  return "msg_" + Math.random().toString(36).slice(2, 11);
}

type ContentBlock = { type: string; text?: string };
type AnthropicMessage = { role: string; content: string | ContentBlock[] };

function convertAnthropicMessages(messages: AnthropicMessage[]): OpenAIMessage[] {
  return messages.map((msg) => {
    const role = msg.role as "user" | "assistant" | "system";
    if (typeof msg.content === "string") {
      return { role, content: msg.content };
    }
    const textParts = (msg.content as ContentBlock[])
      .filter((block) => block.type === "text")
      .map((block) => block.text ?? "");
    return { role, content: textParts.join("\n") };
  });
}

router.options("/v1/messages", (req, res): void => {
  const headers = getCorsHeaders();
  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }
  res.sendStatus(204);
});

router.post("/v1/messages", async (req, res): Promise<void> => {
  const corsHeaders = getCorsHeaders();
  for (const [key, value] of Object.entries(corsHeaders)) {
    res.setHeader(key, value);
  }

  const apiKey = req.headers["x-api-key"] ?? req.headers["authorization"];
  if (!apiKey) {
    res.status(401).json({
      type: "error",
      error: { type: "authentication_error", message: "Missing x-api-key or Authorization header" },
    });
    return;
  }

  const body = req.body;
  if (!body || typeof body !== "object") {
    res.status(400).json({
      type: "error",
      error: { type: "invalid_request_error", message: "Invalid JSON body" },
    });
    return;
  }

  const model: string = body.model ?? "claude-opus-4-6";
  const rawMessages: AnthropicMessage[] = body.messages ?? [];
  const maxTokens: number | undefined = body.max_tokens;
  const temperature: number | undefined = body.temperature;
  const streamRequested: boolean = body.stream === true;

  req.log.info({ model, stream: streamRequested }, "POST /v1/messages");

  const openaiMessages = convertAnthropicMessages(rawMessages);
  const id = generateMsgId();

  if (streamRequested) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    try {
      await callAIStreamAnthropic(model, openaiMessages, { temperature, maxTokens }, res, id);
      req.log.info({ model }, "Stream completed");
    } catch (err) {
      req.log.error({ err, model }, "AI stream failed");
      if (!res.headersSent) {
        res.status(500).json({ type: "error", error: { type: "api_error", message: "AI model call failed" } });
      } else {
        res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message: "Stream error" } })}\n\n`);
        res.end();
      }
    }
    return;
  }

  let content: string;
  try {
    content = await callAI(model, openaiMessages, { temperature, maxTokens });
    req.log.info({ contentLength: content.length }, "Received AI response");
  } catch (err) {
    req.log.error({ err, model }, "AI call failed");
    res.status(500).json({ type: "error", error: { type: "api_error", message: "AI model call failed" } });
    return;
  }

  res.json({
    id,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: content }],
    model,
    stop_reason: "end_turn",
    usage: { input_tokens: 0, output_tokens: 0 },
  });
});

export default router;
