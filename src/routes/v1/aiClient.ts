import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import type { Response } from "express";

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
        blocks.push({
          type: "image",
          source: { type: "url", url },
        });
      }
    }
  }
  return blocks;
}

function sseWrite(res: Response, data: string): void {
  res.write(`data: ${data}\n\n`);
}

// Non-streaming: return full text
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
    const systemText = systemMessages.map((m) => contentToString(m.content)).join("\n");

    const response = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 4096,
      ...(systemText ? { system: systemText } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      messages: nonSystemMessages.map((m) => ({
        role: m.role as "user" | "assistant",
        content: convertContentForAnthropic(m.content),
      })),
    });

    const block = response.content[0];
    return block?.type === "text" ? block.text : "";
  }

  const client = getOpenAIClient();
  const response = await client.chat.completions.create({
    model: canonicalModel,
    messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
  });

  return response.choices[0]?.message?.content ?? "";
}

// Streaming in Anthropic SSE format for /v1/messages endpoint
export async function callAIStreamAnthropic(
  requestedModel: string,
  messages: OpenAIMessage[],
  options: { temperature?: number; maxTokens?: number },
  res: Response,
  id: string,
): Promise<void> {
  const { family, canonicalModel } = resolveModel(requestedModel);

  res.write(
    `event: message_start\ndata: ${JSON.stringify({
      type: "message_start",
      message: { id, type: "message", role: "assistant", content: [], model: requestedModel },
    })}\n\n`,
  );
  res.write(
    `event: content_block_start\ndata: ${JSON.stringify({
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    })}\n\n`,
  );

  if (family === "anthropic") {
    const client = getAnthropicClient();
    const systemMessages = messages.filter((m) => m.role === "system");
    const nonSystemMessages = messages.filter((m) => m.role !== "system");
    const systemText = systemMessages.map((m) => contentToString(m.content)).join("\n");

    const stream = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 4096,
      ...(systemText ? { system: systemText } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      messages: nonSystemMessages.map((m) => ({
        role: m.role as "user" | "assistant",
        content: convertContentForAnthropic(m.content),
      })),
      stream: true,
    });

    for await (const event of stream) {
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        res.write(
          `event: content_block_delta\ndata: ${JSON.stringify({
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: event.delta.text },
          })}\n\n`,
        );
      }
    }
  } else {
    const client = getOpenAIClient();
    const stream = await client.chat.completions.create({
      model: canonicalModel,
      messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      stream: true,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta?.content) {
        res.write(
          `event: content_block_delta\ndata: ${JSON.stringify({
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: delta.content },
          })}\n\n`,
        );
      }
    }
  }

  res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
  res.write(`event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 0 } })}\n\n`);
  res.write(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
  res.end();
}

// Streaming: write SSE chunks directly to res, then end
export async function callAIStream(
  requestedModel: string,
  messages: OpenAIMessage[],
  options: { temperature?: number; maxTokens?: number },
  res: Response,
  id: string,
  created: number,
): Promise<void> {
  const { family, canonicalModel } = resolveModel(requestedModel);

  // Send role chunk first
  sseWrite(
    res,
    JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created,
      model: requestedModel,
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
    }),
  );

  if (family === "anthropic") {
    const client = getAnthropicClient();
    const systemMessages = messages.filter((m) => m.role === "system");
    const nonSystemMessages = messages.filter((m) => m.role !== "system");
    const systemText = systemMessages.map((m) => contentToString(m.content)).join("\n");

    const stream = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 4096,
      ...(systemText ? { system: systemText } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      messages: nonSystemMessages.map((m) => ({
        role: m.role as "user" | "assistant",
        content: convertContentForAnthropic(m.content),
      })),
      stream: true,
    });

    for await (const event of stream) {
      if (
        event.type === "content_block_delta" &&
        event.delta.type === "text_delta"
      ) {
        sseWrite(
          res,
          JSON.stringify({
            id,
            object: "chat.completion.chunk",
            created,
            model: requestedModel,
            choices: [
              {
                index: 0,
                delta: { content: event.delta.text },
                finish_reason: null,
              },
            ],
          }),
        );
      }
    }
  } else {
    const client = getOpenAIClient();
    const stream = await client.chat.completions.create({
      model: canonicalModel,
      messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      stream: true,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta?.content) {
        sseWrite(
          res,
          JSON.stringify({
            id,
            object: "chat.completion.chunk",
            created,
            model: requestedModel,
            choices: [
              {
                index: 0,
                delta: { content: delta.content },
                finish_reason: null,
              },
            ],
          }),
        );
      }
    }
  }

  // Final chunk with finish_reason
  sseWrite(
    res,
    JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created,
      model: requestedModel,
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    }),
  );
  res.write("data: [DONE]\n\n");
  res.end();
}
