import os
import torch
import gc
import json
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
import torchvision.transforms as transforms

# --- 0. 설정 ---
BASE_DIR = os.getcwd()
RESULT_DIR = "/data/jameskimh/homework/continual_CE"
os.makedirs(RESULT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLE_FULL = 100  # 메인 테이블: CS/FID 신뢰도 향상 (20→100)
NUM_SAMPLE_STEP = 8    # step-by-step: CS 추세 파악 중심
NUM_SAMPLE_ABL  = 6    # ablation: 빠른 sweep


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
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

# --- 1. 공통 임베딩 추출 ---

def get_text_embed(pipeline, prompt):
    """Non-special 토큰 평균으로 텍스트 임베딩 추출.
    'Van Gogh' 같은 다중 토큰 개념 올바르게 처리.
    빈 문자열("")은 EOS 위치 임베딩 fallback.
    """
    tokenizer, text_encoder = pipeline.tokenizer, pipeline.text_encoder
    tokens = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=77
    ).input_ids.to(pipeline.device)
    embeds = text_encoder(tokens)[0]  # [1, 77, 768]
    tok = tokens[0]
    bos_id = tokenizer.bos_token_id or 49406
    eos_id = tokenizer.eos_token_id or 49407
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    mask = ~((tok == bos_id) | (tok == eos_id) | (tok == pad_id))
    if mask.sum() > 0:
        return embeds[0][mask].mean(dim=0).to(dtype=torch.float16)
    eos_pos = (tok == eos_id).nonzero(as_tuple=True)[0]
    pos = eos_pos[0].item() if len(eos_pos) > 0 else 1
    return embeds[0, pos].to(dtype=torch.float16)

# --- 2. 알고리즘 구현 ---

def apply_uce_erasure(pipeline, concept_to_erase, preserve_concepts, lamb=0.1):
    """UCE: closed-form cross-attention value projection 수정 (baseline).
    순차 적용 시 이전 개념이 복원되는 catastrophic forgetting 문제 존재.
    """
    c_erase   = get_text_embed(pipeline, concept_to_erase)
    c_pres    = torch.stack([get_text_embed(pipeline, p) for p in preserve_concepts])
    v_anchor  = get_text_embed(pipeline, "")

    for name, module in pipeline.unet.named_modules():
        if module.__class__.__name__ == "Attention" and "attn2" in name:
            W_v = module.to_v.weight.data
            cd, ct = W_v.device, W_v.dtype
            K   = torch.cat([c_erase.unsqueeze(0), c_pres], dim=0).T.to(cd, ct)
            anc = v_anchor.to(cd, ct)
            pe  = c_pres.to(cd, ct)
            Vmat = torch.cat(
                [(W_v @ anc.unsqueeze(1)).squeeze().unsqueeze(1), W_v @ pe.T], dim=1
            ).to(cd, ct)
            KT  = K.T
            inv = torch.inverse(
                (K @ KT).float() + lamb * torch.eye(K.shape[0], device=cd)
            ).to(ct)
            module.to_v.weight.data = (Vmat.float() @ KT.float() @ inv.float()).to(ct)
    return pipeline


def apply_uce_batch_erasure(pipeline, erase_concepts, preserved_dict, lamb=0.1):
    """UCE-Batch: 모든 소거 개념을 단일 closed-form으로 동시 처리 (upper bound).

    Sequential 방법들과 달리 forgetting이 원천적으로 발생하지 않음.
    K = [c_e1, c_e2, c_e3, 모든 보존 개념들...]
    V = [W_v c_*, W_v c_*, W_v c_*, W_v c_p1, ...]
    W_new = V K^T (K K^T + λI)^{-1}

    역할: Sequential 방법들의 ceiling — Batch vs Sequential 격차로
    순차 소거의 본질적 어려움 또는 UCE-EWC의 Batch 근접도를 평가.
    """
    v_anchor = get_text_embed(pipeline, "")

    # 모든 소거 개념 임베딩
    erase_embeds = [get_text_embed(pipeline, c) for c in erase_concepts]

    # 모든 보존 개념 임베딩 (중복 제거)
    all_preserve = []
    seen = set()
    for concept in erase_concepts:
        for p in preserved_dict[concept]:
            if p not in seen:
                all_preserve.append(get_text_embed(pipeline, p))
                seen.add(p)

    for name, module in pipeline.unet.named_modules():
        if module.__class__.__name__ == "Attention" and "attn2" in name:
            W_v = module.to_v.weight.data
            cd, ct = W_v.device, W_v.dtype
            anc = v_anchor.to(cd, ct)

            # K: [소거 개념들 + 보존 개념들]을 열로 쌓음
            all_k = erase_embeds + all_preserve
            K = torch.stack(all_k).to(cd, ct).T  # [d_k, n_erase + n_preserve]

            # V: 소거 개념은 모두 anchor로, 보존 개념은 현재 W_v 출력 유지
            v_erase_cols = [(W_v @ anc.unsqueeze(1)).squeeze().unsqueeze(1)
                            for _ in erase_embeds]
            v_pres_cols  = W_v @ torch.stack(all_preserve).to(cd, ct).T  # [d_v, n_pres]
            Vmat = torch.cat(v_erase_cols + [v_pres_cols], dim=1).to(cd, ct)

            KT  = K.T
            inv = torch.inverse(
                (K @ KT).float() + lamb * torch.eye(K.shape[0], device=cd)
            ).to(ct)
            module.to_v.weight.data = (Vmat.float() @ KT.float() @ inv.float()).to(ct)
    return pipeline


def get_attn2_weights(pipeline):
    """현재 attn2 to_v weight 스냅샷 (CPU 저장)."""
    return {
        name: module.to_v.weight.data.clone().cpu()
        for name, module in pipeline.unet.named_modules()
        if module.__class__.__name__ == "Attention" and "attn2" in name
    }


def compute_importance(prev_w, curr_w):
    """Weight 변화량 제곱 = EWC 스타일 parameter importance."""
    return {
        name: (curr_w[name].float() - prev_w[name].float()) ** 2
        for name in prev_w if name in curr_w
    }


def accumulate_importance(imp_a, imp_b):
    """여러 step의 importance 누적."""
    if imp_a is None:
        return imp_b
    all_names = set(imp_a) | set(imp_b)
    return {
        name: imp_a.get(name, torch.zeros(1)) + imp_b.get(name, torch.zeros(1))
        for name in all_names
    }


def apply_uce_ewc_erasure(pipeline, concept_to_erase, preserve_concepts,
                           importance_dict=None, alpha=1.0, lamb=0.1):
    """UCE-EWC: EWC 스타일 importance 정규화로 catastrophic forgetting 완화 (Ours).
    W_new = V @ K^T @ (K @ K^T + λI + α·diag(F))^{-1}
    F = 이전 erasure에서 변화량이 큰 weight 방향 → 보호 강도 부여
    """
    c_erase  = get_text_embed(pipeline, concept_to_erase)
    c_pres   = torch.stack([get_text_embed(pipeline, p) for p in preserve_concepts])
    v_anchor = get_text_embed(pipeline, "")

    for name, module in pipeline.unet.named_modules():
        if module.__class__.__name__ == "Attention" and "attn2" in name:
            W_v = module.to_v.weight.data
            cd, ct = W_v.device, W_v.dtype
            K  = torch.cat([c_erase.unsqueeze(0), c_pres], dim=0).T.to(cd, ct)
            anc = v_anchor.to(cd, ct)
            pe  = c_pres.to(cd, ct)
            Vmat = torch.cat(
                [(W_v @ anc.unsqueeze(1)).squeeze().unsqueeze(1), W_v @ pe.T], dim=1
            ).to(cd, ct)
            KT = K.T
            d_k = K.shape[0]

            reg = lamb * torch.eye(d_k, device=cd, dtype=torch.float32)
            if importance_dict is not None and name in importance_dict:
                imp   = importance_dict[name].to(cd)
                f_imp = imp.mean(dim=0).float()
                f_max = f_imp.max()
                if f_max > 1e-12:
                    f_imp = f_imp / f_max
                reg = reg + alpha * torch.diag(f_imp)

            inv  = torch.inverse((K @ KT).float() + reg).to(ct)
            module.to_v.weight.data = (Vmat.float() @ KT.float() @ inv.float()).to(ct)
    return pipeline

# --- 3. 평가 함수 ---

def evaluate_model(pipeline, prompts, clip_model, clip_processor, fid_metric,
                   save_dir, real_images=None):
    """이미지 생성 후 CLIP Score & FID 계산.
    이미 N개 이미지가 존재하면 생성을 skip하고 기존 파일 로드.
    Returns: (cs_score, fid_score, fake_stack[N,3,512,512] uint8)
    """
    from PIL import Image
    pipeline.set_progress_bar_config(disable=True)
    generated, tensors = [], []
    to_tensor = transforms.ToTensor()
    out_dir = os.path.join(RESULT_DIR, save_dir)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    n_existing = len(existing)

    def _load_or_regen(idx, prompt):
        """기존 파일 로드 시도; 깨진 파일이면 재생성."""
        path = os.path.join(out_dir, f"sample_{idx}.png")
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                img.load()
                return img
            except Exception:
                print(f"  [warn] truncated {path}, regenerating")
        img = pipeline(prompt, num_inference_steps=30).images[0]
        img.save(path)
        return img

    # 이미 충분한 이미지가 있으면 로드만, 아니면 resume
    if n_existing >= len(prompts):
        print(f"  [skip] {save_dir} — {n_existing} images already exist")
    elif n_existing > 0:
        print(f"  [resume] {save_dir} — {n_existing}/{len(prompts)} done, generating rest")

    for i, prompt in enumerate(prompts):
        if i < n_existing:
            img = _load_or_regen(i, prompt)
        else:
            img = pipeline(prompt, num_inference_steps=30).images[0]
            img.save(os.path.join(out_dir, f"sample_{i}.png"))
        generated.append(img)
        tensors.append((to_tensor(img.resize((512, 512))) * 255).to(torch.uint8))

    inputs = clip_processor(
        text=prompts, images=generated, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        cs = (image_embeds * text_embeds).sum(dim=-1).mean().item()

    fake = torch.stack(tensors).to(device)
    fid  = 0.0
    if real_images is not None:
        try:
            fid_metric.update(fake, real=False)
            fid_metric.update(real_images.to(device), real=True)
            fid = fid_metric.compute().item()
        except Exception as e:
            print(f"  [Warning] FID failed for {save_dir}: {e}")
        finally:
            fid_metric.reset()

    return cs, fid, fake


def save_visual_samples(pipeline, concepts, prefix="Final"):
    pipeline.set_progress_bar_config(disable=True)
    print(f"  >> Visual samples: {prefix}")
    save_dir = os.path.join(RESULT_DIR, prefix, "visual_samples")
    os.makedirs(save_dir, exist_ok=True)
    for concept in concepts:
        prompt = (f"A painting in the style of {concept}"
                  if "Gogh" in concept else f"A high quality photo of {concept}")
        csafe = concept.replace(" ", "_")
        for i in range(3):
            img = pipeline(prompt, num_inference_steps=30).images[0]
            img.save(os.path.join(save_dir, f"{csafe}_sample_{i}.png"))


def load_coco_real_images(n=500):
    """실제 이미지 n개를 로드하여 [N,3,512,512] uint8 텐서 반환.
    FID의 real image reference로 사용.
    phiyodr/coco2017 시도 후 실패 시 nlphuji/flickr30k로 fallback.
    """
    import io
    from PIL import Image as PILImage
    to_tensor = transforms.ToTensor()

    def _decode_image(img_data):
        if isinstance(img_data, PILImage.Image):
            return img_data.convert("RGB")
        if isinstance(img_data, bytes):
            return PILImage.open(io.BytesIO(img_data)).convert("RGB")
        if isinstance(img_data, dict):
            b = img_data.get("bytes")
            if b:
                return PILImage.open(io.BytesIO(b)).convert("RGB")
        return None

    def _collect(dataset_name, split, img_key, n):
        ds = load_dataset(dataset_name, split=split, streaming=True)
        tensors = []
        for item in ds:
            if len(tensors) >= n:
                break
            try:
                img = _decode_image(item.get(img_key))
                if img is None:
                    continue
                img = img.resize((512, 512))
                tensors.append((to_tensor(img) * 255).to(torch.uint8))
            except Exception:
                continue
        return tensors

    print("  Trying phiyodr/coco2017...")
    tensors = _collect("phiyodr/coco2017", "validation", "image", n)

    if len(tensors) == 0:
        print("  [warn] phiyodr/coco2017 failed, falling back to nlphuji/flickr30k")
        tensors = _collect("nlphuji/flickr30k", "test", "image", n)

    if len(tensors) == 0:
        raise RuntimeError("Could not load any real images for FID reference")

    print(f"  Loaded {len(tensors)} real images for FID reference")
    return torch.stack(tensors)


def get_eval_prompts(concept, n):
    if "Gogh" in concept:
        return [f"A painting in the style of {concept}"] * n
    return [f"A photo of {concept}"] * n

# --- 4. Step-by-step 중간 평가 헬퍼 ---

def eval_erased_so_far(pipe, erased_concepts, preserved_dict,
                       clip_model, clip_processor, fid_metric,
                       prefix, reference_images):
    """지금까지 지운 개념들을 평가 → forgetting curve 데이터 수집.
    FID는 reference 있는 경우에만, CS는 항상 계산.
    """
    results = {}
    for concept in erased_concepts:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_STEP)
        cs_e, fid_e, _ = evaluate_model(
            pipe, prompts_e, clip_model, clip_processor, fid_metric,
            f"{prefix}/{concept.replace(' ', '_')}_e",
            real_images=reference_images.get(f"{concept}_e")
        )
        results[f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
    return results

# --- 5. 실험 함수 ---

def run_experiment(quantitative_table, forgetting_curve, reference_images,
                   clip_model, clip_processor, fid_metric,
                   coco_prompts, coco_real_images, target_concepts, preserved_dict):
    """3단계 실험 + step-by-step forgetting curve."""

    # [Stage 0] Original SD v1.4 — FID reference 수집
    print("\n[Stage 0] Original SD v1.4 (FID reference collection)...")
    pipe = setup_pipeline()
    for concept in target_concepts:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_FULL)
        csafe = concept.replace(" ", "_")
        cs_e, _, ref_e = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                         fid_metric, f"SD_v1_4/{csafe}_e")
        cs_r, _, ref_r = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                         clip_processor, fid_metric, f"SD_v1_4/{csafe}_r")
        reference_images[f"{concept}_e"] = ref_e.cpu()
        reference_images[f"{concept}_r"] = ref_r.cpu()
        quantitative_table["SD_v1_4"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": 0.0}
        quantitative_table["SD_v1_4"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": 0.0}

    cs_coco, fid_coco, _ = evaluate_model(pipe, coco_prompts, clip_model, clip_processor,
                                           fid_metric, "SD_v1_4/coco",
                                           real_images=coco_real_images)
    quantitative_table["SD_v1_4"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    save_visual_samples(pipe, target_concepts, "SD_v1_4")
    del pipe; clear_memory()

    # [Stage 1] UCE Baseline + step-by-step
    print("\n[Stage 1] UCE Baseline (sequential + step-by-step)...")
    pipe = setup_pipeline()
    erased_so_far = []
    forgetting_curve["UCE"] = {}

    for concept in target_concepts:
        pipe = apply_uce_erasure(pipe, concept, preserved_dict[concept])
        erased_so_far.append(concept)
        step_key = f"after_{concept}"
        print(f"  [UCE step] Erased {erased_so_far} — evaluating forgetting...")
        forgetting_curve["UCE"][step_key] = eval_erased_so_far(
            pipe, erased_so_far, preserved_dict,
            clip_model, clip_processor, fid_metric,
            f"UCE/step/after_{concept.replace(' ', '_')}", reference_images
        )

    for concept in target_concepts:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_FULL)
        csafe = concept.replace(" ", "_")
        cs_e, fid_e, _ = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                         fid_metric, f"UCE/{csafe}_e",
                                         real_images=reference_images[f"{concept}_e"])
        cs_r, fid_r, _ = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                         clip_processor, fid_metric, f"UCE/{csafe}_r",
                                         real_images=reference_images[f"{concept}_r"])
        quantitative_table["UCE_baseline"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
        quantitative_table["UCE_baseline"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": round(fid_r, 4)}

    cs_coco, fid_coco, _ = evaluate_model(pipe, coco_prompts, clip_model, clip_processor,
                                           fid_metric, "UCE/coco",
                                           real_images=coco_real_images)
    quantitative_table["UCE_baseline"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    save_visual_samples(pipe, target_concepts, "UCE")
    del pipe; clear_memory()

    # [Stage 2] UCE-EWC (Ours) + step-by-step
    print("\n[Stage 2] UCE-EWC Ours (sequential + step-by-step)...")
    pipe = setup_pipeline()
    accumulated_imp = None
    erased_so_far   = []
    forgetting_curve["UCE_EWC"] = {}

    for concept in target_concepts:
        prev_w = get_attn2_weights(pipe)
        pipe   = apply_uce_ewc_erasure(pipe, concept, preserved_dict[concept],
                                        importance_dict=accumulated_imp, alpha=1.0)
        curr_w = get_attn2_weights(pipe)
        accumulated_imp = accumulate_importance(accumulated_imp,
                                                compute_importance(prev_w, curr_w))
        erased_so_far.append(concept)
        step_key = f"after_{concept}"
        print(f"  [UCE-EWC step] Erased {erased_so_far} — evaluating forgetting...")
        forgetting_curve["UCE_EWC"][step_key] = eval_erased_so_far(
            pipe, erased_so_far, preserved_dict,
            clip_model, clip_processor, fid_metric,
            f"UCE_EWC/step/after_{concept.replace(' ', '_')}", reference_images
        )

    for concept in target_concepts:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_FULL)
        csafe = concept.replace(" ", "_")
        cs_e, fid_e, _ = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                         fid_metric, f"UCE_EWC/{csafe}_e",
                                         real_images=reference_images[f"{concept}_e"])
        cs_r, fid_r, _ = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                         clip_processor, fid_metric, f"UCE_EWC/{csafe}_r",
                                         real_images=reference_images[f"{concept}_r"])
        quantitative_table["UCE_EWC_ours"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
        quantitative_table["UCE_EWC_ours"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": round(fid_r, 4)}

    cs_coco, fid_coco, _ = evaluate_model(pipe, coco_prompts, clip_model, clip_processor,
                                           fid_metric, "UCE_EWC/coco",
                                           real_images=coco_real_images)
    quantitative_table["UCE_EWC_ours"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    save_visual_samples(pipe, target_concepts, "UCE_EWC")
    del pipe; clear_memory()

    # [Stage 3] UCE-Batch (upper bound)
    print("\n[Stage 3] UCE-Batch (single-shot simultaneous erasure, upper bound)...")
    pipe = setup_pipeline()
    pipe = apply_uce_batch_erasure(pipe, target_concepts, preserved_dict)

    for concept in target_concepts:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_FULL)
        csafe = concept.replace(" ", "_")
        cs_e, fid_e, _ = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                         fid_metric, f"UCE_Batch/{csafe}_e",
                                         real_images=reference_images[f"{concept}_e"])
        cs_r, fid_r, _ = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                         clip_processor, fid_metric, f"UCE_Batch/{csafe}_r",
                                         real_images=reference_images[f"{concept}_r"])
        quantitative_table["UCE_Batch"][f"{concept}_e"] = {"CS": round(cs_e, 4), "FID": round(fid_e, 4)}
        quantitative_table["UCE_Batch"][f"{concept}_r"] = {"CS": round(cs_r, 4), "FID": round(fid_r, 4)}

    cs_coco, fid_coco, _ = evaluate_model(pipe, coco_prompts, clip_model, clip_processor,
                                           fid_metric, "UCE_Batch/coco",
                                           real_images=coco_real_images)
    quantitative_table["UCE_Batch"]["MS-COCO"] = {"CS": round(cs_coco, 4), "FID": round(fid_coco, 4)}
    save_visual_samples(pipe, target_concepts, "UCE_Batch")
    del pipe; clear_memory()


def run_ablation_alpha(clip_model, clip_processor, fid_metric,
                       reference_images, coco_prompts, target_concepts, preserved_dict):
    """Ablation 1: alpha sweep + preservation CS 포함 (Van Gogh_r 역전 현상 분석)."""
    print("\n[Ablation 1] UCE-EWC alpha sweep (erasure + preservation)...")
    alpha_values = [0.01, 0.1, 1.0, 10.0]
    results = {}

    for alpha in alpha_values:
        print(f"  >> alpha={alpha}...")
        pipe = setup_pipeline()
        accumulated_imp = None
        for concept in target_concepts:
            prev_w = get_attn2_weights(pipe)
            pipe   = apply_uce_ewc_erasure(pipe, concept, preserved_dict[concept],
                                            importance_dict=accumulated_imp, alpha=alpha)
            curr_w = get_attn2_weights(pipe)
            accumulated_imp = accumulate_importance(accumulated_imp,
                                                    compute_importance(prev_w, curr_w))
        row = {}
        for concept in target_concepts:
            # erased 개념 CS
            prompts_e = get_eval_prompts(concept, NUM_SAMPLE_ABL)
            csafe = concept.replace(" ", "_")
            cs_e, _, _ = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                         fid_metric, f"ablation_alpha/alpha_{alpha}/{csafe}_e")
            row[f"{concept}_e_CS"] = round(cs_e, 4)
            # preserved 개념 CS (Van Gogh_r 역전 분석)
            cs_r, _, _ = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                         clip_processor, fid_metric, f"ablation_alpha/alpha_{alpha}/{csafe}_r")
            row[f"{concept}_r_CS"] = round(cs_r, 4)

        cs_coco, _, _ = evaluate_model(pipe, coco_prompts[:20], clip_model, clip_processor,
                                        fid_metric, f"ablation_alpha/alpha_{alpha}/coco")
        row["MS-COCO_CS"] = round(cs_coco, 4)
        results[f"alpha_{alpha}"] = row
        del pipe; clear_memory()

    return results


def run_order_ablation(clip_model, clip_processor, fid_metric,
                       reference_images, coco_prompts, target_concepts, preserved_dict):
    """Ablation 2: 소거 순서 민감도 — 역순 (Snoopy→Van Gogh→Superman).
    UCE와 UCE-EWC 각각 실험. step-by-step forgetting curve 포함.
    """
    print("\n[Ablation 2] Erasure order sensitivity (Snoopy→Van Gogh→Superman)...")
    reversed_order = list(reversed(target_concepts))  # Snoopy, Van Gogh, Superman
    order_results  = {"UCE": {}, "UCE_EWC": {}}
    order_forgetting = {"UCE": {}, "UCE_EWC": {}}

    # UCE reversed
    print("  >> UCE reversed order...")
    pipe = setup_pipeline()
    erased_so_far = []
    for concept in reversed_order:
        pipe = apply_uce_erasure(pipe, concept, preserved_dict[concept])
        erased_so_far.append(concept)
        order_forgetting["UCE"][f"after_{concept}"] = eval_erased_so_far(
            pipe, erased_so_far, preserved_dict,
            clip_model, clip_processor, fid_metric,
            f"ablation_order/UCE/step/after_{concept.replace(' ', '_')}", reference_images
        )
    for concept in reversed_order:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_ABL)
        csafe = concept.replace(" ", "_")
        cs_e, _, _ = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                     fid_metric, f"ablation_order/UCE/final/{csafe}_e")
        cs_r, _, _ = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                     clip_processor, fid_metric, f"ablation_order/UCE/final/{csafe}_r")
        order_results["UCE"][f"{concept}_e_CS"] = round(cs_e, 4)
        order_results["UCE"][f"{concept}_r_CS"] = round(cs_r, 4)
    cs_c, _, _ = evaluate_model(pipe, coco_prompts[:20], clip_model, clip_processor,
                                 fid_metric, "ablation_order/UCE/coco")
    order_results["UCE"]["MS-COCO_CS"] = round(cs_c, 4)
    del pipe; clear_memory()

    # UCE-EWC reversed
    print("  >> UCE-EWC reversed order...")
    pipe = setup_pipeline()
    accumulated_imp = None
    erased_so_far   = []
    for concept in reversed_order:
        prev_w = get_attn2_weights(pipe)
        pipe   = apply_uce_ewc_erasure(pipe, concept, preserved_dict[concept],
                                        importance_dict=accumulated_imp, alpha=1.0)
        curr_w = get_attn2_weights(pipe)
        accumulated_imp = accumulate_importance(accumulated_imp,
                                                compute_importance(prev_w, curr_w))
        erased_so_far.append(concept)
        order_forgetting["UCE_EWC"][f"after_{concept}"] = eval_erased_so_far(
            pipe, erased_so_far, preserved_dict,
            clip_model, clip_processor, fid_metric,
            f"ablation_order/UCE_EWC/step/after_{concept.replace(' ', '_')}", reference_images
        )
    for concept in reversed_order:
        prompts_e = get_eval_prompts(concept, NUM_SAMPLE_ABL)
        csafe = concept.replace(" ", "_")
        cs_e, _, _ = evaluate_model(pipe, prompts_e, clip_model, clip_processor,
                                     fid_metric, f"ablation_order/UCE_EWC/final/{csafe}_e")
        cs_r, _, _ = evaluate_model(pipe, preserved_dict[concept], clip_model,
                                     clip_processor, fid_metric, f"ablation_order/UCE_EWC/final/{csafe}_r")
        order_results["UCE_EWC"][f"{concept}_e_CS"] = round(cs_e, 4)
        order_results["UCE_EWC"][f"{concept}_r_CS"] = round(cs_r, 4)
    cs_c, _, _ = evaluate_model(pipe, coco_prompts[:20], clip_model, clip_processor,
                                 fid_metric, "ablation_order/UCE_EWC/coco")
    order_results["UCE_EWC"]["MS-COCO_CS"] = round(cs_c, 4)
    del pipe; clear_memory()

    return order_results, order_forgetting

# --- 6. 메인 ---

def main():
    target_concepts = ["Superman", "Van Gogh", "Snoopy"]
    preserved_dict  = {
        "Superman": ["Batman", "Thor", "Wonder Woman", "Shazam"],
        "Van Gogh": ["Picasso", "Monet", "Paul Gauguin", "Caravaggio"],
        "Snoopy":   ["Mickey", "Spongebob", "Pikachu", "Hello Kitty"],
    }

    clear_memory()
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", use_safetensors=True
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch16", use_fast=False
    )
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    coco_ds = load_dataset("sentence-transformers/coco-captions", split="train")
    print(f"Dataset columns: {coco_ds.column_names}")
    if "caption" in coco_ds.column_names:
        key = "caption"
    elif "text" in coco_ds.column_names:
        key = "text"
    else:
        key = coco_ds.column_names[0]
    coco_prompts = [coco_ds[i][key] for i in range(500)]
    print(f"Loaded 500 COCO prompts (key='{key}')")

    print("Loading real COCO images for FID reference...")
    coco_real_images = load_coco_real_images(n=500).to(device)

    quantitative_table = {
        "SD_v1_4": {}, "UCE_baseline": {}, "UCE_EWC_ours": {}, "UCE_Batch": {}
    }
    forgetting_curve = {}
    reference_images = {}

    # ── 메인 실험 (step-by-step 포함) ──
    run_experiment(
        quantitative_table, forgetting_curve, reference_images,
        clip_model, clip_processor, fid_metric,
        coco_prompts, coco_real_images, target_concepts, preserved_dict
    )

    # ── Ablation 1: alpha sweep (erasure + preservation CS) ──
    ablation_alpha = run_ablation_alpha(
        clip_model, clip_processor, fid_metric,
        reference_images, coco_prompts, target_concepts, preserved_dict
    )

    # ── Ablation 2: 소거 순서 민감도 ──
    order_results, order_forgetting = run_order_ablation(
        clip_model, clip_processor, fid_metric,
        reference_images, coco_prompts, target_concepts, preserved_dict
    )

    # ── 전체 결과 저장 ──
    output = {
        "quantitative_table":  quantitative_table,
        "forgetting_curve":    forgetting_curve,    # step-by-step CS
        "ablation_alpha":      ablation_alpha,       # alpha sweep (erasure+preservation)
        "ablation_order": {
            "order":           "Snoopy→Van Gogh→Superman",
            "final_table":     order_results,
            "forgetting_curve": order_forgetting,
        },
    }
    with open(os.path.join("/data/jameskimh/homework/continual_CE", "quantitative_table.json"), "w") as f:
        json.dump(output, f, indent=4)
    print("\n>> All Results Saved to ./results/quantitative_table.json")


if __name__ == "__main__":
    main()
