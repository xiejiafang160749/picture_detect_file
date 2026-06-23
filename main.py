import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

import httpx

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

DETECTION_PROMPT = """You are an expert forensic image analyst specializing in detecting AI-generated images.

Analyze this image carefully for signs of AI generation. Look for:
1. Unnatural textures, patterns, or repetitive structures
2. Inconsistent lighting, shadows, or reflections
3. Facial abnormalities (distorted ears, asymmetrical features, unusual eyes, extra fingers)
4. Blurry or morphed backgrounds, especially at edges
5. Impossible physics or geometry
6. Overly smooth skin or textures lacking natural variation
7. Watermark artifacts or model-specific patterns
8. Unnatural bokeh or depth-of-field effects
9. Text or lettering that appears garbled or nonsensical
10. Inconsistencies between foreground and background elements

Based on your analysis, provide your assessment in the following JSON format ONLY, with no additional text:
{
  "probability": <number between 0 and 100, where 0=real photo, 100=AI-generated>,
  "confidence": "<low|medium|high>",
  "key_indicators": ["<indicator 1>", "<indicator 2>", ...],
  "reasoning": "<brief explanation>"
}

Where probability=0 means definitely real photo, probability=100 means definitely AI-generated."""

AI_THRESHOLD = 0.5  # 判断为AI生成的概率阈值（0~1），大于等于此值则判定为AI生成

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


async def detect_image(client: httpx.AsyncClient, image_path: Path, api_key: str) -> dict:
    image_data = image_path.read_bytes()
    mime_type = MIME_TYPES[image_path.suffix.lower()]
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": DETECTION_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = await client.post(
        f"{SILICONFLOW_BASE_URL}/chat/completions",
        json=payload,
        headers=headers
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()

    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if not json_match:
        raise ValueError(f"模型返回格式异常: {content}")

    detection = json.loads(json_match.group())
    probability = float(detection.get("probability", 50))
    probability = max(0.0, min(100.0, probability))
    probability_decimal = round(probability / 100, 4)

    return {
        "filename": image_path.name,
        "is_ai_generated": probability_decimal >= AI_THRESHOLD,
        "probability": probability_decimal,
        "confidence": detection.get("confidence", "medium"),
        "key_indicators": detection.get("key_indicators", []),
        "reasoning": detection.get("reasoning", "")
    }


async def process_folder(input_folder: Path, output_folder: Path, api_key: str):
    images = [
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print(f"未找到支持的图片文件（支持格式：{', '.join(SUPPORTED_EXTENSIONS)}）")
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(images)} 张图片，开始检测...\n")

    summary = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for image_path in images:
            print(f"处理: {image_path.name} ...", end=" ", flush=True)
            try:
                result = await detect_image(client, image_path, api_key)
                # Save individual result
                out_file = output_folder / (image_path.stem + ".json")
                out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                summary.append(result)
                print(f"AI概率: {result['probability']} | 判断: {'AI生成' if result['is_ai_generated'] else '真实照片'} [{result['confidence']}]")
            except Exception as e:
                error_result = {"filename": image_path.name, "error": str(e)}
                out_file = output_folder / (image_path.stem + ".json")
                out_file.write_text(json.dumps(error_result, ensure_ascii=False, indent=2), encoding="utf-8")
                summary.append(error_result)
                print(f"失败: {e}")

    # Save summary
    summary_file = output_folder / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！结果已保存到: {output_folder}")
    print(f"汇总文件: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="AI图片检测批处理工具")
    parser.add_argument("input_folder", nargs="?", default="example-s6", help="输入图片文件夹（默认: example-s6）")
    parser.add_argument("output_folder", nargs="?", default="2026ai-result", help="输出结果文件夹（默认: 2026ai-result）")
    parser.add_argument("--api-key", default=os.environ.get("SILICONFLOW_API_KEY"), help="SiliconFlow API Key（也可通过环境变量 SILICONFLOW_API_KEY 设置）")
    args = parser.parse_args()

    if not args.api_key:
        print("错误：未提供 API Key，请通过 --api-key 参数或 SILICONFLOW_API_KEY 环境变量传入")
        sys.exit(1)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.exists():
        print(f"错误：输入文件夹不存在: {input_folder}")
        sys.exit(1)

    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}\n")

    asyncio.run(process_folder(input_folder, output_folder, args.api_key))


if __name__ == "__main__":
    main()
