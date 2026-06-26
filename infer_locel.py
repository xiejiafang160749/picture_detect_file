import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

import httpx

LOCAL_BASE_URL = "http://localhost:8000/v1"
VISION_MODEL = "/data/models/Qwen3-VL-32B-Instruct"

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

AI_THRESHOLD = 0.5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


async def detect_image(client: httpx.AsyncClient, image_path: Path) -> dict:
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

    response = await client.post(
        f"{LOCAL_BASE_URL}/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
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


async def process_folder(input_folder: Path, output_folder: Path):
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        for image_path in images:
            print(f"处理: {image_path.name} ...", end=" ", flush=True)
            try:
                result = await detect_image(client, image_path)
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

    summary_file = output_folder / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成提交格式 JSONL：image_name + is_generated ("1"=AI生成, "0"=真实)
    jsonl_file = output_folder / "result.jsonl"
    with jsonl_file.open("w", encoding="utf-8") as f:
        for r in summary:
            if "error" not in r:
                f.write(json.dumps({
                    "image_name": r["filename"],
                    "is_generated": "1" if r["is_ai_generated"] else "0"
                }, ensure_ascii=False) + "\n")

    print(f"\n完成！结果已保存到: {output_folder}")
    print(f"汇总文件: {summary_file}")
    print(f"提交文件: {jsonl_file}")


def main():
    global LOCAL_BASE_URL, VISION_MODEL
    parser = argparse.ArgumentParser(description="本地模型 AI 图片检测批处理工具")
    parser.add_argument("input_folder", nargs="?", default="example-s6", help="输入图片文件夹（默认: example-s6）")
    parser.add_argument("output_folder", nargs="?", default="2026ai-result", help="输出结果文件夹（默认: 2026ai-result）")
    parser.add_argument("--base-url", default=os.environ.get("LOCAL_MODEL_URL", LOCAL_BASE_URL), help="本地模型服务地址（默认: http://localhost:8000/v1）")
    parser.add_argument("--model", default=VISION_MODEL, help=f"模型名称（默认: {VISION_MODEL}）")
    args = parser.parse_args()

    LOCAL_BASE_URL = args.base_url
    VISION_MODEL = args.model

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.exists():
        print(f"错误：输入文件夹不存在: {input_folder}")
        sys.exit(1)

    print(f"本地模型地址: {LOCAL_BASE_URL}")
    print(f"使用模型: {VISION_MODEL}")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}\n")

    asyncio.run(process_folder(input_folder, output_folder))


if __name__ == "__main__":
    main()
