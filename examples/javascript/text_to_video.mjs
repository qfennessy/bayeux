/**
 * Text-to-video with fetch.
 * Note: endpoint is /v1/video/ (no 's'), not /v1/videos/.
 *
 * Usage:
 *   NUNCHAKU_API_KEY=sk-nunchaku-... node text_to_video.mjs
 */

import { writeFileSync } from "fs";

const API_KEY = process.env.NUNCHAKU_API_KEY;
if (!API_KEY) {
  console.error("Set NUNCHAKU_API_KEY environment variable.");
  process.exit(1);
}

console.log("Generating video (this may take ~30 seconds)...");

const response = await fetch(
  "https://api.nunchaku.dev/v1/video/generations",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "nunchaku-wan2.2-lightning-t2v",
      prompt: "A golden retriever running on a beach at sunset, cinematic",
      n: 1,
      size: "1280x720",
      num_frames: 81,
      num_inference_steps: 4,
      guidance_scale: 1.0,
      response_format: "b64_json",
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
writeFileSync("output_t2v.mp4", buf);
console.log(
  `Saved output_t2v.mp4 (${(buf.length / 1024 / 1024).toFixed(1)} MB)`
);
