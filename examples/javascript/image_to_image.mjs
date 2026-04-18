/**
 * Image-to-image (edit) with fetch.
 *
 * Input image is passed via the `url` field as a data URI — not a file upload.
 *
 * Usage:
 *   NUNCHAKU_API_KEY=sk-nunchaku-... node image_to_image.mjs input.jpg "make it a watercolor"
 */

import { readFileSync, writeFileSync } from "fs";

const API_KEY = process.env.NUNCHAKU_API_KEY;
if (!API_KEY) {
  console.error("Set NUNCHAKU_API_KEY environment variable.");
  process.exit(1);
}

const inputPath = process.argv[2];
if (!inputPath) {
  console.error("Usage: node image_to_image.mjs <input_image> [prompt]");
  process.exit(1);
}

const prompt =
  process.argv[3] || "transform this into a watercolor painting";

const imageBytes = readFileSync(inputPath);
const imageB64 = imageBytes.toString("base64");
const mime = inputPath.endsWith(".png") ? "image/png" : "image/jpeg";
const dataUri = `data:${mime};base64,${imageB64}`;

const response = await fetch(
  "https://api.nunchaku.dev/v1/images/edits",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "nunchaku-qwen-image-edit",
      prompt,
      url: dataUri,
      n: 1,
      size: "1024x1024",
      tier: "fast",
      num_inference_steps: 28,
      response_format: "b64_json",
    }),
  }
);

if (!response.ok) {
  console.error(`Error ${response.status}: ${await response.text()}`);
  process.exit(1);
}

const data = await response.json();
const buf = Buffer.from(data.data[0].b64_json, "base64");
writeFileSync("output_i2i.jpg", buf);
console.log(`Saved output_i2i.jpg (${buf.length.toLocaleString()} bytes)`);
