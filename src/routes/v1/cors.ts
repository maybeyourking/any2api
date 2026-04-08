export function getCorsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers":
      "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta",
    "Access-Control-Max-Age": "86400",
  };
}
