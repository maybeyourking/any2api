const MODEL_MAP: Record<string, string> = {
  "gpt-5": "/integrations/chat-gpt/conversationgpt4",
  "gpt-5-turbo": "/integrations/chat-gpt/conversationgpt4",
  "chat-gpt": "/integrations/chat-gpt/conversationgpt4",

  "claude-opus-4.6": "/integrations/anthropic-claude-opus-4-6",
  "claude-opus-4-6": "/integrations/anthropic-claude-opus-4-6",
  "anthropic-claude-opus-4-6": "/integrations/anthropic-claude-opus-4-6",

  "claude-opus-4.1": "/integrations/anthropic-claude-opus-4-1",
  "claude-opus-4-1": "/integrations/anthropic-claude-opus-4-1",
  "anthropic-claude-opus-4-1": "/integrations/anthropic-claude-opus-4-1",

  "claude-sonnet-4.5": "/integrations/anthropic-claude-sonnet-4-5",
  "claude-sonnet-4-5": "/integrations/anthropic-claude-sonnet-4-5",
  "claude-4.6-sonnet": "/integrations/anthropic-claude-sonnet-4-5",
  "claude-4-6-sonnet": "/integrations/anthropic-claude-sonnet-4-5",
  "anthropic-claude-sonnet-4-5": "/integrations/anthropic-claude-sonnet-4-5",

  "claude-sonnet-4": "/integrations/anthropic-claude-sonnet-4",
  "anthropic-claude-sonnet-4": "/integrations/anthropic-claude-sonnet-4",

  "gemini-2.5-pro": "/integrations/google-gemini-2-5-pro",
  "gemini-2-5-pro": "/integrations/google-gemini-2-5-pro",
  "google-gemini-2-5-pro": "/integrations/google-gemini-2-5-pro",

  "gemini-2.5-flash": "/integrations/google-gemini-2-5-flash",
  "gemini-2-5-flash": "/integrations/google-gemini-2-5-flash",
  "google-gemini-2-5-flash": "/integrations/google-gemini-2-5-flash",

  "gemini-3.0-pro": "/integrations/google-gemini-3-0-pro",
  "gemini-3-0-pro": "/integrations/google-gemini-3-0-pro",
  "google-gemini-3-0-pro": "/integrations/google-gemini-3-0-pro",

  o3: "/integrations/openai-o3",
  "openai-o3": "/integrations/openai-o3",

  "grok-4": "/integrations/grok-4-0709",

  "gpt-oss": "/integrations/gpt-oss",
  "gpt-oss-120b": "/integrations/gpt-oss",
};

const FALLBACK_PATH = "/integrations/chat-gpt/conversationgpt4";

export function resolveIntegrationPath(model: string): string {
  return MODEL_MAP[model] ?? FALLBACK_PATH;
}
