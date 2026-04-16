/**
 * Image-to-Video — animate a static image into a video.
 *
 * Endpoint: POST /v1/video/animations   (note: /video/ not /videos/)
 * Model:    nunchaku-wan2.2-lightning-i2v
 *
 * IMPORTANT: The input image is passed inside `messages` using multimodal
 * content blocks — NOT via a simple `image` field.
 *
 * Usage:
 *   NUNCHAKU_API_KEY=sk-nunchaku-... node image_to_video.mjs input.jpg
 */

import { readFileSync, writeFileSync } from "fs";

const API_KEY = process.env.NUNCHAKU_API_KEY;
if (!API_KEY) {
  console.error("Set NUNCHAKU_API_KEY environment variable.");
  process.exit(1);
}

const inputPath = process.argv[2];
if (!inputPath) {
  console.error("Usage: node image_to_video.mjs <input_image> [prompt]");
  process.exit(1);
}

const prompt =
  process.argv[3] || "the scene comes to life with gentle motion";

// Encode input image as data URI
const imageBytes = readFileSync(inputPath);
const imageB64 = imageBytes.toString("base64");
const mime = inputPath.endsWith(".png") ? "image/png" : "image/jpeg";
const dataUri = `data:${mime};base64,${imageB64}`;

console.log("Generating video (this may take ~30 seconds)...");

const response = await fetch(
  "https://api.nunchaku.dev/v1/video/animations",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "nunchaku-wan2.2-lightning-i2v",
      prompt,
      n: 1,
      size: "1280x720",
      num_frames: 81,
      num_inference_steps: 4,
      guidance_scale: 1.0,
      response_format: "b64_json",
      // Image MUST be inside `messages` with multimodal content blocks
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: dataUri } },
            { type: "text", text: prompt },
          ],
        },
      ],
    }),
    signal: AbortSignal.timeout(120_000),
  }
);

if (!response.ok) {
  console.error(`Error ${response.status}: ${await response.text()}`);
  process.exit(1);
}

const data = await response.json();
const buf = Buffer.from(data.data[0].b64_json, "base64");
writeFileSync("output_i2v.mp4", buf);
console.log(
  `Saved output_i2v.mp4 (${(buf.length / 1024 / 1024).toFixed(1)} MB)`
);
