/**
 * Text-to-Image — generate an image from a text prompt.
 *
 * Endpoint: POST /v1/images/generations
 * Model:    nunchaku-qwen-image
 *
 * Usage:
 *   NUNCHAKU_API_KEY=sk-nunchaku-... node text_to_image.mjs
 */

import { writeFileSync } from "fs";

const API_KEY = process.env.NUNCHAKU_API_KEY;
if (!API_KEY) {
  console.error("Set NUNCHAKU_API_KEY environment variable.");
  process.exit(1);
}

const response = await fetch(
  "https://api.nunchaku.dev/v1/images/generations",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "nunchaku-qwen-image",
      prompt:
        "a red apple on a wooden table, photorealistic, studio lighting",
      n: 1,
      size: "1024x1024",
      tier: "fast",
      num_inference_steps: 28,
      response_format: "b64_json",
      seed: 42,
    }),
  }
);

if (!response.ok) {
  console.error(`Error ${response.status}: ${await response.text()}`);
  process.exit(1);
}

const data = await response.json();
const buf = Buffer.from(data.data[0].b64_json, "base64");
writeFileSync("output_t2i.jpg", buf);
console.log(`Saved output_t2i.jpg (${buf.length.toLocaleString()} bytes)`);
