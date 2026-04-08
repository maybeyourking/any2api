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

// ── Types ────────────────────────────────────────────────────────────────────

type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } };

type OpenAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

export type OpenAIMessage = {
  role: "user" | "assistant" | "system" | "tool";
  content: string | OpenAIContentPart[] | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  name?: string;
};

export type OpenAITool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: object;
  };
};

export type CallOptions = {
  temperature?: number;
  maxTokens?: number;
  tools?: OpenAITool[];
  toolChoice?: unknown;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function contentToString(content: string | OpenAIContentPart[] | null): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p) => p.type === "text")
    .map((p) => (p as { type: "text"; text: string }).text)
    .join("\n");
}

function convertContentForAnthropic(
  content: string | OpenAIContentPart[] | null,
): Anthropic.MessageParam["content"] {
  if (!content) return "";
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
        blocks.push({ type: "image", source: { type: "url", url } });
      }
    }
  }
  return blocks;
}

function convertToolsToAnthropic(tools: OpenAITool[]): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
    input_schema: (t.function.parameters ?? {
      type: "object",
      properties: {},
    }) as Anthropic.Tool["input_schema"],
  }));
}

function convertMessagesToAnthropic(
  messages: OpenAIMessage[],
): { system: string; messages: Anthropic.MessageParam[] } {
  const systemParts: string[] = [];
  const result: Anthropic.MessageParam[] = [];

  let i = 0;
  while (i < messages.length) {
    const msg = messages[i];

    if (msg.role === "system") {
      systemParts.push(contentToString(msg.content));
      i++;
      continue;
    }

    if (msg.role === "assistant") {
      const contentBlocks: Anthropic.ContentBlockParam[] = [];
      const textStr = contentToString(msg.content);
      if (textStr) {
        contentBlocks.push({ type: "text", text: textStr });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let input: object = {};
          try { input = JSON.parse(tc.function.arguments || "{}"); } catch { /* ignore */ }
          contentBlocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input,
          });
        }
      }
      result.push({
        role: "assistant",
        content: contentBlocks.length > 0 ? contentBlocks : contentToString(msg.content),
      });
      i++;
      continue;
    }

    if (msg.role === "tool") {
      // Collect consecutive tool results into a single user message
      const toolResults: Anthropic.ContentBlockParam[] = [];
      while (i < messages.length && messages[i].role === "tool") {
        const t = messages[i];
        toolResults.push({
          type: "tool_result",
          tool_use_id: t.tool_call_id ?? "",
          content: contentToString(t.content),
        });
        i++;
      }
      result.push({ role: "user", content: toolResults });
      continue;
    }

    // user
    result.push({
      role: "user",
      content: convertContentForAnthropic(msg.content),
    });
    i++;
  }

  return { system: systemParts.join("\n"), messages: result };
}

function anthropicResponseToOpenAI(
  response: Anthropic.Message,
  id: string,
  created: number,
  model: string,
) {
  const textBlocks = response.content.filter(
    (b): b is Anthropic.TextBlock => b.type === "text",
  );
  const toolUseBlocks = response.content.filter(
    (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
  );

  const textContent = textBlocks.map((b) => b.text).join("") || null;

  if (toolUseBlocks.length > 0) {
    return {
      id,
      object: "chat.completion",
      created,
      model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: textContent,
            tool_calls: toolUseBlocks.map((b) => ({
              id: b.id,
              type: "function",
              function: {
                name: b.name,
                arguments: JSON.stringify(b.input),
              },
            })),
          },
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
  }

  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: textContent ?? "" },
        finish_reason: response.stop_reason === "end_turn" ? "stop" : (response.stop_reason ?? "stop"),
      },
    ],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

// ── Non-streaming call (returns OpenAI response object) ──────────────────────

export async function callAIResponse(
  requestedModel: string,
  messages: OpenAIMessage[],
  options: CallOptions,
  id: string,
  created: number,
): Promise<object> {
  const { family, canonicalModel } = resolveModel(requestedModel);

  if (family === "anthropic") {
    const client = getAnthropicClient();
    const { system, messages: anthropicMessages } = convertMessagesToAnthropic(messages);

    const response = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 8096,
      ...(system ? { system } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      ...(options.tools ? { tools: convertToolsToAnthropic(options.tools) } : {}),
      messages: anthropicMessages,
    });

    return anthropicResponseToOpenAI(response, id, created, requestedModel);
  }

  const client = getOpenAIClient();
  const response = await client.chat.completions.create({
    model: canonicalModel,
    messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.tools ? { tools: options.tools as OpenAI.Chat.ChatCompletionTool[] } : {}),
    ...(options.toolChoice !== undefined ? { tool_choice: options.toolChoice as OpenAI.Chat.ChatCompletionToolChoiceOption } : {}),
  });

  return response;
}

// Legacy helper for non-tool text-only responses
export async function callAI(
  requestedModel: string,
  messages: OpenAIMessage[],
  options: { temperature?: number; maxTokens?: number } = {},
): Promise<string> {
  const { family, canonicalModel } = resolveModel(requestedModel);

  if (family === "anthropic") {
    const client = getAnthropicClient();
    const { system, messages: anthropicMessages } = convertMessagesToAnthropic(messages);

    const response = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 8096,
      ...(system ? { system } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      messages: anthropicMessages,
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

// ── Streaming (Anthropic SSE format) for /v1/messages ────────────────────────

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
    const { system, messages: anthropicMessages } = convertMessagesToAnthropic(messages);

    const stream = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 8096,
      ...(system ? { system } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      messages: anthropicMessages,
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

// ── Streaming (OpenAI SSE format) for /v1/chat/completions ───────────────────

function sseWrite(res: Response, data: string): void {
  res.write(`data: ${data}\n\n`);
}

export async function callAIStream(
  requestedModel: string,
  messages: OpenAIMessage[],
  options: CallOptions,
  res: Response,
  id: string,
  created: number,
): Promise<void> {
  const { family, canonicalModel } = resolveModel(requestedModel);

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
    const { system, messages: anthropicMessages } = convertMessagesToAnthropic(messages);

    // Accumulate tool_use blocks while streaming text
    const toolUseMap: Map<number, { id: string; name: string; inputStr: string }> = new Map();

    const stream = await client.messages.create({
      model: canonicalModel,
      max_tokens: options.maxTokens ?? 8096,
      ...(system ? { system } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature as number } : {}),
      ...(options.tools ? { tools: convertToolsToAnthropic(options.tools) } : {}),
      messages: anthropicMessages,
      stream: true,
    });

    let stopReason = "stop";

    for await (const event of stream) {
      if (event.type === "content_block_start" && event.content_block.type === "tool_use") {
        toolUseMap.set(event.index, {
          id: event.content_block.id,
          name: event.content_block.name,
          inputStr: "",
        });
      } else if (event.type === "content_block_delta") {
        if (event.delta.type === "text_delta") {
          sseWrite(
            res,
            JSON.stringify({
              id,
              object: "chat.completion.chunk",
              created,
              model: requestedModel,
              choices: [{ index: 0, delta: { content: event.delta.text }, finish_reason: null }],
            }),
          );
        } else if (event.delta.type === "input_json_delta") {
          const tool = toolUseMap.get(event.index);
          if (tool) tool.inputStr += event.delta.partial_json;
        }
      } else if (event.type === "message_delta") {
        if (event.delta.stop_reason === "tool_use") stopReason = "tool_calls";
      }
    }

    // Send accumulated tool_calls as a single chunk
    if (toolUseMap.size > 0) {
      const toolCalls = Array.from(toolUseMap.entries()).map(([idx, t]) => ({
        index: idx,
        id: t.id,
        type: "function",
        function: { name: t.name, arguments: t.inputStr },
      }));
      sseWrite(
        res,
        JSON.stringify({
          id,
          object: "chat.completion.chunk",
          created,
          model: requestedModel,
          choices: [{ index: 0, delta: { tool_calls: toolCalls }, finish_reason: null }],
        }),
      );
    }

    sseWrite(
      res,
      JSON.stringify({
        id,
        object: "chat.completion.chunk",
        created,
        model: requestedModel,
        choices: [{ index: 0, delta: {}, finish_reason: stopReason }],
      }),
    );
  } else {
    const client = getOpenAIClient();
    const stream = await client.chat.completions.create({
      model: canonicalModel,
      messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.tools ? { tools: options.tools as OpenAI.Chat.ChatCompletionTool[] } : {}),
      ...(options.toolChoice !== undefined ? { tool_choice: options.toolChoice as OpenAI.Chat.ChatCompletionToolChoiceOption } : {}),
      stream: true,
    });

    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      if (!choice) continue;
      const delta = choice.delta;
      if (delta.content || delta.tool_calls || choice.finish_reason) {
        sseWrite(
          res,
          JSON.stringify({
            id,
            object: "chat.completion.chunk",
            created,
            model: requestedModel,
            choices: [{ index: 0, delta, finish_reason: choice.finish_reason ?? null }],
          }),
        );
      }
    }
  }

  res.write("data: [DONE]\n\n");
  res.end();
}
