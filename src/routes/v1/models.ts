import { Router, type IRouter } from "express";
import { getCorsHeaders } from "./cors";

const router: IRouter = Router();

const CREATED_AT = 1712275200;
const OWNED_BY = "anything";

const MODEL_IDS = [
  "gpt-5",
  "gpt-5-turbo",
  "chat-gpt",
  "claude-opus-4.6",
  "claude-opus-4-6",
  "anthropic-claude-opus-4-6",
  "claude-opus-4.1",
  "claude-opus-4-1",
  "anthropic-claude-opus-4-1",
  "claude-sonnet-4.5",
  "claude-sonnet-4-5",
  "claude-4.6-sonnet",
  "claude-4-6-sonnet",
  "anthropic-claude-sonnet-4-5",
  "claude-sonnet-4",
  "anthropic-claude-sonnet-4",
  "gemini-2.5-pro",
  "gemini-2-5-pro",
  "google-gemini-2-5-pro",
  "gemini-2.5-flash",
  "gemini-2-5-flash",
  "google-gemini-2-5-flash",
  "gemini-3.0-pro",
  "gemini-3-0-pro",
  "google-gemini-3-0-pro",
  "o3",
  "openai-o3",
  "grok-4",
  "gpt-oss",
  "gpt-oss-120b",
];

router.options("/v1/models", (req, res): void => {
  const headers = getCorsHeaders();
  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }
  res.sendStatus(204);
});

router.get("/v1/models", (req, res): void => {
  const headers = getCorsHeaders();
  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }

  req.log.info("GET /v1/models — listing all available models");

  const data = MODEL_IDS.map((id) => ({
    id,
    object: "model",
    created: CREATED_AT,
    owned_by: OWNED_BY,
  }));

  res.json({ object: "list", data });
});

export default router;
