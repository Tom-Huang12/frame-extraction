import os
import cv2
import numpy as np
from PIL import Image
import imagehash
import torch
import open_clip


# =========================
# 初始化 CLIP
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def calc_hash(img):
    return imagehash.phash(img)


def is_duplicate(hash_list, new_hash, threshold=6):
    for h in hash_list:
        if abs(h - new_hash) <= threshold:
            return True
    return False


# =========================
# CLIP评分（核心优化版）
# =========================
def clip_score(img):
    text = "a good video frame"

    image_tensor = preprocess(img).unsqueeze(0).to(device)
    text_tensor = tokenizer([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (image_features @ text_features.T).item()

    return score


# =========================
# 差分函数
# =========================
def frame_diff(f1, f2):
    diff = cv2.absdiff(f1, f2)
    return np.mean(diff)


# =========================
# 核心：关键帧提取
# =========================
def extract_keyframes(
    video_path,
    output_dir="output/frames",
    interval_sec=1,
    diff_threshold=15,
    clip_threshold=0.15,
    phash_threshold=6
):

    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    interval = int(fps * interval_sec)

    last_frame = None
    hashes = []
    saved = []

    idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =========================
        # 1. 定时抽帧（保证稳定输出）
        # =========================
        if idx % interval != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =========================
        # 2. Diff过滤（去静态画面）
        # =========================
        if last_frame is not None:
            diff = frame_diff(gray, last_frame)
            if diff < diff_threshold:
                idx += 1
                continue

        last_frame = gray

        # =========================
        # 3. 转PIL
        # =========================
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # =========================
        # 4. CLIP语义过滤
        # =========================
        score = clip_score(img)
        if score < clip_threshold:
            idx += 1
            continue

        # =========================
        # 5. PHash去重
        # =========================
        h = calc_hash(img)
        if is_duplicate(hashes, h, phash_threshold):
            idx += 1
            continue

        hashes.append(h)

        # =========================
        # 6. 保存
        # =========================
        filename = os.path.join(output_dir, f"frame_{save_idx:04d}.jpg")
        img.save(filename)
        saved.append(filename)

        save_idx += 1
        idx += 1

    cap.release()
    return saved


# =========================
# 拼图
# =========================
def merge_frames(image_list, output_path="output/summary.jpg", cols=5):
    if not image_list:
        print("没有关键帧")
        return

    images = [Image.open(img) for img in image_list]

    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols

    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))

    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        canvas.paste(img, (x, y))

    canvas.save(output_path)


# =========================
# 自动命名
# =========================
def auto_name(frames):
    n = len(frames)

    if n > 30:
        return "rich_scene"
    elif n > 10:
        return "medium_scene"
    else:
        return "simple_scene"


# =========================
# 主函数
# =========================
def main():
    print("=== AI Video Keyframe Pipeline ===")

    video_path = input("请输入视频路径：").strip()

    if not os.path.exists(video_path):
        print("视频不存在")
        return

    base_dir = "output"
    frames_dir = os.path.join(base_dir, "frames")

    print("\n[1] 提取关键帧...")
    frames = extract_keyframes(
        video_path,
        output_dir=frames_dir,
        interval_sec=1,
        diff_threshold=15,
        clip_threshold=0.15
    )

    print(f"关键帧数量: {len(frames)}")

    print("\n[2] 拼接缩略图...")
    merge_frames(
        frames,
        os.path.join(base_dir, "summary.jpg")
    )

    print("\n[3] 自动命名...")
    tag = auto_name(frames)

    print(f"视频标签: {tag}")

    print("\n完成输出目录：output/")


if __name__ == "__main__":
    main()