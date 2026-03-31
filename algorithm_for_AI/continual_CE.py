import os
import torch
import gc
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
import torchvision.transforms as transforms

# --- 1. 환경 설정 및 메모리 최적화 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def setup_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4" # Backbone: SD v1.4 [cite: 20]
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None, 
        requires_safety_checker=False
    )
    # 메모리 최적화 옵션
    pipe.enable_model_cpu_offload() 
    pipe.enable_attention_slicing()
    return pipe

# --- 2. UCE (Baseline) 알고리즘 구현 ---
def apply_uce_erasure(pipeline, concept_to_erase, preserve_concepts, lamb=0.1):
    """UCE 알고리즘을 통한 가중치 업데이트 [cite: 31, 36]"""
    print(f"\n[UCE] Erasing '{concept_to_erase}' while preserving {preserve_concepts}...")
    
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device)
    
    def get_embed(prompt):
        tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device)
        return text_encoder(tokens)[0][0, 1]

    c_erase = get_embed(concept_to_erase)
    c_preserves = torch.stack([get_embed(p) for p in preserve_concepts])
    v_anchor_embed = get_embed("")

    for name, module in pipeline.unet.named_modules():
        if module.__class__.__name__ == "Attention" and "attn2" in name:
            W_v = module.to_v.weight.data
            K = torch.cat([c_erase.unsqueeze(0), c_preserves], dim=0).T
            
            target_v_erase = (W_v @ v_anchor_embed.unsqueeze(1)).squeeze()
            target_v_preserves = (W_v @ c_preserves.T)
            V = torch.cat([target_v_erase.unsqueeze(1), target_v_preserves], dim=1)
            
            KT = K.T
            inv_part = torch.inverse(K @ KT + lamb * torch.eye(K.shape[0]).to(device))
            W_new = V @ KT @ inv_part
            module.to_v.weight.data = W_new.to(torch.float16)
    
    return pipeline

# --- 3. 평가 함수 (CLIP Score & FID) ---
def evaluate_model(pipeline, prompts, clip_model, clip_processor, fid_metric, real_images=None):
    pipeline.set_progress_bar_config(disable=True)
    generated_images = []
    to_tensor = transforms.ToTensor()
    
    for prompt in prompts:
        img = pipeline(prompt, num_inference_steps=30).images[0]
        generated_images.append(img)
        
        # FID용 텐서 변환
        img_tensor = (to_tensor(img.resize((512, 512))) * 255).to(torch.uint8).to(device)
        fid_metric.update(img_tensor.unsqueeze(0), real=False)
    
    # CLIP Score (CS)
    inputs = clip_processor(text=prompts, images=generated_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    cs_score = outputs.logits_per_image.diag().mean().item()
    
    # FID
    if real_images is not None:
        fid_metric.update(real_images, real=True)
    fid_score = fid_metric.compute().item()
    fid_metric.reset()
    
    return cs_score, fid_score

# --- 4. 메인 실험 루프 ---
def main():
    clear_memory()
    pipe = setup_pipeline()
    
    # 평가 지표 초기화
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # 데이터 설정 [cite: 21, 40, 41, 42]
    target_concepts = ["Superman", "Van Gogh", "Snoopy"]
    preserved_dict = {
        "Superman_r": ["Batman", "Thor", "Wonder Woman", "Shazam"],
        "Van Gogh_r": ["Picasso", "Monet", "Paul Gauguin", "Caravaggio"],
        "Snoopy_r": ["Mickey", "Spongebob", "Pikachu", "Hello Kitty"]
    }

    # MS-COCO 로드
    print("Loading MS-COCO dataset...")
    coco_ds = load_dataset("detection-datasets/coco", split="val", trust_remote_code=True)
    coco_prompts = [coco_ds[i]['caption'] for i in range(50)]
    
    # 실제 이미지 준비 (FID용)
    real_imgs = []
    for i in range(50):
        img = coco_ds[i]['image'].convert("RGB").resize((512, 512))
        real_imgs.append((transforms.ToTensor()(img) * 255).to(torch.uint8))
    real_imgs_tensor = torch.stack(real_imgs).to(device)

    # 결과 저장을 위한 테이블 
    final_results = []

    # Sequential Erasure 수행 [cite: 8, 12]
    for concept in target_concepts:
        # 1. 지우기 전 Baseline 측정 (Optional: 첫 루프에서만 수행)
        
        # 2. 삭제 수행
        pipe = apply_uce_erasure(pipe, concept, preserved_dict[f"{concept}_r"])
        
        # 3. 현재 지운 개념 평가 (Target CS)
        cs, fid = evaluate_model(pipe, [f"A photo of {concept}"] * 10, clip_model, clip_processor, fid_metric)
        print(f">> Result for {concept}: CS={cs:.4f}, FID={fid:.4f}")
        final_results.append({"Concept": concept, "CS": cs, "FID": fid})

    # 4. 최종 MS-COCO 평가 
    coco_cs, coco_fid = evaluate_model(pipe, coco_prompts, clip_model, clip_processor, fid_metric, real_images=real_imgs_tensor)
    print(f"\nFinal MS-COCO: CS={coco_cs:.4f}, FID={coco_fid:.4f}")
    
    print("\n[Done] All concepts erased sequentially.")

if __name__ == "__main__":
    main()