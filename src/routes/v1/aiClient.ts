import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

function getOpenAIClient(): OpenAI {
  const baseURL = process.env.AI_INTEGRATIONS_OPENAI_BASE_URL;
  const apiKey = process.env.AI_INTEGRATIONS_OPENAI_API_KEY ?? "dummy";
  return new OpenAI({ baseURL, apiKey });
}

function getAnthropicClient(): Anthropic {
  const baseURL = process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL;
  const apiKey = process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY ?? "dummy";
  return new Anthropic({ baseURL, apiKey });
}

export type ModelFamily = "openai" | "anthropic";

const ANTHROPIC_ALIASES: Set<string> = new Set([
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
]);

const MODEL_CANONICAL: Record<string, string> = {
  "claude-opus-4.6": "claude-opus-4-6",
  "claude-opus-4-6": "claude-opus-4-6",
  "anthropic-claude-opus-4-6": "claude-opus-4-6",
  "claude-opus-4.1": "claude-opus-4-1",
  "claude-opus-4-1": "claude-opus-4-1",
  "anthropic-claude-opus-4-1": "claude-opus-4-1",
  "claude-sonnet-4.5": "claude-sonnet-4-5",
  "claude-sonnet-4-5": "claude-sonnet-4-5",
  "claude-4.6-sonnet": "claude-sonnet-4-5",
  "claude-4-6-sonnet": "claude-sonnet-4-5",
  "anthropic-claude-sonnet-4-5": "claude-sonnet-4-5",
  "claude-sonnet-4": "claude-sonnet-4-5",
  "anthropic-claude-sonnet-4": "claude-sonnet-4-5",
  "gemini-2.5-pro": "gpt-5",
  "gemini-2-5-pro": "gpt-5",
  "google-gemini-2-5-pro": "gpt-5",
  "gemini-2.5-flash": "gpt-5",
  "gemini-2-5-flash": "gpt-5",
  "google-gemini-2-5-flash": "gpt-5",
  "gemini-3.0-pro": "gpt-5",
  "gemini-3-0-pro": "gpt-5",
  "google-gemini-3-0-pro": "gpt-5",
  o3: "o3",
  "openai-o3": "o3",
  "grok-4": "gpt-5",
  "gpt-oss": "gpt-5",
  "gpt-oss-120b": "gpt-5",
};

export function resolveModel(requestedModel: string): {
  family: ModelFamily;
  canonicalModel: string;
} {
  if (ANTHROPIC_ALIASES.has(requestedModel)) {
    const canonical = MODEL_CANONICAL[requestedModel] ?? "claude-sonnet-4-5";
    return { family: "anthropic", canonicalModel: canonical };
  }
  const canonical = MODEL_CANONICAL[requestedModel] ?? "gpt-5";
  return { family: "openai", canonicalModel: canonical };
}

type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } };

export type OpenAIMessage = {
  role: "user" | "assistant" | "system";
  content: string | OpenAIContentPart[];
};

function contentToString(content: string | OpenAIContentPart[]): string {
  if (typeof content === "string") return content;
  return content
    .filter((p) => p.type === "text")
    .map((p) => (p as { type: "text"; text: string }).text)
    .join("\n");
}

function convertContentForAnthropic(
  content: string | OpenAIContentPart[],
): Anthropic.MessageParam["content"] {
  if (typeof content === "string") return content;

  const blocks: Anthropic.ContentBlockParam[] = [];
  for (const part of content) {
    if (part.type === "text") {
      blocks.push({ type: "text", text: part.text });
    } else if (part.type === "image_url") {
      const url = part.image_url.url;
      if (url.startsWith("data:")) {
        // base64 data URL: data:<media_type>;base64,<data>
        const match = url.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          blocks.push({
            type: "image",
            source: {
              type: "base64",
              media_type: match[1] as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
              data: match[2],
            },
          });
        }
      } else {
        // Regular URL
        blocks.push({
          type: "image",
          source: {
            type: "url",
            url,
          },
        });
      }
    }
  }
  return blocks;
}

export async function callAI(
  requestedModel: string,
  messages: OpenAIMessage[],
  options: { temperature?: number; maxTokens?: number } = {},
): Promise<string> {
  const { family, canonicalModel } = resolveModel(requestedModel);

  if (family === "anthropic") {
    const client = getAnthropicClient();
    const systemMessages = messages.filter((m) => m.role === "system");
    const nonSystemMessages = messages.filter((m) => m.role !== "system");
    const systemText = systemMessages
      .map((m) => contentToString(m.content))
      .join("\n");

    const response = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 4096,
      ...(systemText ? { system: systemText } : {}),
      ...(options.temperature !== undefined
        ? { temperature: options.temperature as number }
        : {}),
      messages: nonSystemMessages.map((m) => ({
        role: m.role as "user" | "assistant",
        content: convertContentForAnthropic(m.content),
      })),
    });

    const block = response.content[0];
    if (block?.type === "text") {
      return block.text;
    }
    return "";
  }

  const client = getOpenAIClient();
  const response = await client.chat.completions.create({
    model: canonicalModel,
    messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
    ...(options.temperature !== undefined
      ? { temperature: options.temperature }
      : {}),
  });

  return response.choices[0]?.message?.content ?? "";
}
