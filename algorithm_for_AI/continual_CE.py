import os
import torch
import gc
import numpy as np
import json
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
import torchvision.transforms as transforms

# --- 0. 디렉토리 및 환경 설정 ---
BASE_DIR = os.getcwd() 
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def setup_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    )
    pipe.enable_model_cpu_offload() 
    pipe.enable_attention_slicing()
    return pipe

# --- 1. 알고리즘 구현 (UCE & SCA-UCE) ---

def apply_uce_erasure(pipeline, concept_to_erase, preserve_concepts, lamb=0.1):
    tokenizer, text_encoder = pipeline.tokenizer, pipeline.text_encoder
    def get_embed(prompt):
        tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(pipeline.device)
        return text_encoder(tokens)[0][0, 1].to(dtype=torch.float16)

    c_erase = get_embed(concept_to_erase)
    c_preserves = torch.stack([get_embed(p) for p in preserve_concepts])
    v_anchor = get_embed("")

    for name, module in pipeline.unet.named_modules():
        if module.__class__.__name__ == "Attention" and "attn2" in name:
            W_v = module.to_v.weight.data
            curr_device, curr_dtype = W_v.device, W_v.dtype
            K = torch.cat([c_erase.unsqueeze(0), c_preserves], dim=0).T.to(device=curr_device, dtype=curr_dtype)
            anchor = v_anchor.to(device=curr_device, dtype=curr_dtype)
            pres_embeds = c_preserves.to(device=curr_device, dtype=curr_dtype)
            target_v_erase = (W_v @ anchor.unsqueeze(1)).squeeze()
            target_v_preserves = (W_v @ pres_embeds.T)
            V = torch.cat([target_v_erase.unsqueeze(1), target_v_preserves], dim=1).to(device=curr_device, dtype=curr_dtype)
            KT = K.T
            inv_part = torch.inverse((K @ KT).float() + lamb * torch.eye(K.shape[0], device=curr_device)).to(curr_dtype)
            W_new = (V.float() @ KT.float() @ inv_part.float()).to(curr_dtype)
            module.to_v.weight.data = W_new
    return pipeline

def apply_sca_erasure(pipeline, concept_to_erase, preserve_concepts, past_concepts=None, lamb=0.1):
    if past_concepts is None: past_concepts = []
    tokenizer, text_encoder = pipeline.tokenizer, pipeline.text_encoder
    def get_embed(prompt):
        tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(pipeline.device)
        return text_encoder(tokens)[0][0, 1].to(dtype=torch.float16)

    c_now = get_embed(concept_to_erase)
    c_pasts = [get_embed(c) for c in past_concepts]
    c_preserves = [get_embed(p) for p in preserve_concepts]
    v_anchor = get_embed("")

    for name, module in pipeline.unet.named_modules():
        if module.__class__.__name__ == "Attention" and "attn2" in name:
            W_v = module.to_v.weight.data
            curr_device, curr_dtype = W_v.device, W_v.dtype
            all_k = [c_now] + c_pasts + c_preserves
            K = torch.stack(all_k).T.to(curr_device, curr_dtype)
            target_v_anchor = (W_v @ v_anchor.to(curr_device, curr_dtype).unsqueeze(1)).squeeze()
            pres_embeds = torch.stack(c_preserves).to(curr_device, curr_dtype)
            target_v_preserves = (W_v @ pres_embeds.T)
            v_list = [target_v_anchor.unsqueeze(1)] * (1 + len(c_pasts))
            V = torch.cat(v_list + [target_v_preserves], dim=1).to(curr_device, curr_dtype)
            K_f32, KT_f32 = K.float(), K.T.float()
            inv_part = torch.inverse(K_f32 @ KT_f32 + lamb * torch.eye(K.shape[0], device=curr_device)).to(curr_dtype)
            W_new = (V.float() @ KT_f32 @ inv_part.float()).to(curr_dtype)
            module.to_v.weight.data = W_new
    return pipeline

# --- 2. 평가 및 저장 함수 ---

def evaluate_model(pipeline, prompts, clip_model, clip_processor, fid_metric, concept_name, real_images=None):
    pipeline.set_progress_bar_config(disable=True)
    generated_images = []
    to_tensor = transforms.ToTensor()
    concept_path = os.path.join(RESULT_DIR, concept_name)
    os.makedirs(concept_path, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        img = pipeline(prompt, num_inference_steps=30).images[0]
        generated_images.append(img)
        img.save(os.path.join(concept_path, f"sample_{i}.png"))
        img_tensor = (to_tensor(img.resize((512, 512))) * 255).to(torch.uint8).to(device)
        fid_metric.update(img_tensor.unsqueeze(0), real=False)
    
    inputs = clip_processor(text=prompts, images=generated_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    cs_score = outputs.logits_per_image.diag().mean().item()
    
    fid_score = 0.0
    if real_images is not None:
        fid_metric.update(real_images, real=True)
        fid_score = fid_metric.compute().item()
    fid_metric.reset()
    return cs_score, fid_score

def save_visual_samples(pipeline, concepts, prefix="Final"):
    pipeline.set_progress_bar_config(disable=True)
    print(f">> Generating {prefix} Visual Samples...")
    for concept in concepts:
        prompt = f"A high quality photo of {concept}"
        if "Gogh" in concept: prompt = f"A painting in the style of {concept}"
        for i in range(3):
            image = pipeline(prompt, num_inference_steps=30).images[0]
            image.save(os.path.join(RESULT_DIR, f"{prefix}_{concept}_sample_{i}.png"))

# --- 3. 메인 실험 루프 ---

def main():
    # 데이터 설정
    target_concepts = ["Superman", "Van Gogh", "Snoopy"]
    preserved_dict = {
        "Superman_r": ["Batman", "Thor", "Wonder Woman", "Shazam"],
        "Van Gogh_r": ["Picasso", "Monet", "Paul Gauguin", "Caravaggio"],
        "Snoopy_r": ["Mickey", "Spongebob", "Pikachu", "Hello Kitty"]
    }
    
    # 평가 도구 및 데이터 로드
    clear_memory()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)


    # 1. 데이터셋 로드
    coco_ds = load_dataset("sentence-transformers/coco-captions", split="train")
    
    # 2. 실제 존재하는 컬럼명 확인 (디버깅용)
    print(f"Available columns in dataset: {coco_ds.column_names}")
    
    # 3. 'caption' 키가 없으면 'text'를 찾고, 그것도 없으면 첫 번째 컬럼을 사용
    if 'caption' in coco_ds.column_names:
        actual_key = 'caption'
    elif 'text' in coco_ds.column_names:
        actual_key = 'text'
    else:
        actual_key = coco_ds.column_names[0] # 가장 안전한 방법
    
    # 4. 프롬프트 리스트 생성 (수정된 부분)
    coco_prompts = [coco_ds[i][actual_key] for i in range(50)]
    
    print(f"Successfully loaded 50 prompts using key: '{actual_key}'")

    real_imgs_tensor = torch.zeros((50, 3, 512, 512), dtype=torch.uint8).to(device)

    # 최종 결과 테이블 구조
    quantitative_table = {"SD_v1_4": {}, "UCE_baseline": {}, "SCA_ours": {}}

    NUM_SAMPLE=4

    # [Stage 0] Original SD v1.4
    print("\n[Stage 0] Measuring Original Model...")
    pipe = setup_pipeline()
    for concept in target_concepts:
        cs_e, fid_e = evaluate_model(pipe, [f"A photo of {concept}"] * NUM_SAMPLE, clip_model, clip_processor, fid_metric, "orig_e", real_imgs_tensor)
        cs_r, fid_r = evaluate_model(pipe, preserved_dict[f"{concept}_r"], clip_model, clip_processor, fid_metric, "orig_r", real_imgs_tensor)
        quantitative_table["SD_v1_4"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
        quantitative_table["SD_v1_4"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": round(fid_r, 4)}
    
    cs_coco, fid_coco = evaluate_model(pipe, coco_prompts, clip_model, clip_processor, fid_metric, "orig_coco", real_imgs_tensor)
    quantitative_table["SD_v1_4"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    del pipe; clear_memory()

    # [Stage 1] UCE Baseline
    print("\n[Stage 1] Sequential Erasure (UCE Baseline)...")
    pipe = setup_pipeline()
    for concept in target_concepts:
        pipe = apply_uce_erasure(pipe, concept, preserved_dict[f"{concept}_r"])
    
    for concept in target_concepts:
        cs_e, fid_e = evaluate_model(pipe, [f"A photo of {concept}"] * NUM_SAMPLE, clip_model, clip_processor, fid_metric, "uce_e", real_imgs_tensor)
        cs_r, fid_r = evaluate_model(pipe, preserved_dict[f"{concept}_r"], clip_model, clip_processor, fid_metric, "uce_r", real_imgs_tensor)
        quantitative_table["UCE_baseline"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
        quantitative_table["UCE_baseline"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": round(fid_r, 4)}
    
    cs_coco, fid_coco = evaluate_model(pipe, coco_prompts, clip_model, clip_processor, fid_metric, "uce_coco", real_imgs_tensor)
    quantitative_table["UCE_baseline"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    save_visual_samples(pipe, target_concepts, prefix="UCE")
    del pipe; clear_memory()

    # [Stage 2] SCA-UCE (Ours)
    print("\n[Stage 2] Sequential Erasure (SCA-UCE Ours)...")
    pipe = setup_pipeline()
    erased_history = []
    for concept in target_concepts:
        pipe = apply_sca_erasure(pipe, concept, preserved_dict[f"{concept}_r"], past_concepts=erased_history)
        erased_history.append(concept)
    
    for concept in target_concepts:
        cs_e, fid_e = evaluate_model(pipe, [f"A photo of {concept}"] * NUM_SAMPLE, clip_model, clip_processor, fid_metric, "ours_e", real_imgs_tensor)
        cs_r, fid_r = evaluate_model(pipe, preserved_dict[f"{concept}_r"], clip_model, clip_processor, fid_metric, "ours_r", real_imgs_tensor)
        quantitative_table["SCA_ours"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
        quantitative_table["SCA_ours"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": round(fid_r, 4)}
    
    cs_coco, fid_coco = evaluate_model(pipe, coco_prompts, clip_model, clip_processor, fid_metric, "ours_coco", real_imgs_tensor)
    quantitative_table["SCA_ours"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    save_visual_samples(pipe, target_concepts, prefix="Ours")

    # JSON 저장
    with open(os.path.join(RESULT_DIR, "quantitative_table.json"), "w") as f:
        json.dump(quantitative_table, f, indent=4)
    print("\n>> All Results Saved to ./results/quantitative_table.json")

if __name__ == "__main__":
    main()
