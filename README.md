# TOPS

**First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference**

TOPS greedily combines task relevance, information coverage, and semantic diversity. Stage 1 reduces visual tokens before the language model; stage 2 refines the retained set at designated language-model layers. This repository packages the original implementations and public launchers for seven backbones.

**Release status:** source preview. Syntax and launcher dry runs are checked; full environment installation, GPU inference, and paper-result reproduction remain unverified for this extracted release. The dependency files are candidate pins, not certified lockfiles.

## Model mapping

| Directory | Public model alias | Original implementation ID | `--tokens` |
| --- | --- | --- | --- |
| `LLaVA-TOPS` | `llava-1.5-7b`, `llava-1.5-13b` | `prefixvlm_2` | 128 / 64 / 32 |
| `LLaVA-TOPS` | `llava-next-7b`, `llava-next-13b` | `prefixvlm_2` | 128 / 64 / 32, internally scaled for five views |
| `LLaVA-Video-TOPS` | `llava-video-7b` | `tops` | 64 / 32 / 16 per frame |
| `VLMEvalKit-TOPS` | `qwen2.5-vl-7b`, `internvl3-8b` | `tops` | 256 / 128 |

`prefixvlm_2` is the image LLaVA implementation corresponding to TOPS. The older `prefixvlm`, the S1-only `tops_iclr` and `tops_2`, and other historical methods are not selected by the public launcher. Supporting framework and baseline modules remain where imported by the original code.

For NeXT, pass **128/64/32**, corresponding to nominal total budgets 640/320/160. Do not pass 640/320/160 to this TOPS launcher: the original implementation uses base-budget table lookup and applies its own five-view multiplier. This differs from other pruning methods in the same framework.

## Environments

Use Linux x86_64 and a CUDA-compatible NVIDIA GPU/driver. The image environments match the corresponding MiCo profiles:

| Component | Environment | Python | PyTorch | Transformers |
| --- | --- | --- | --- | --- |
| Image LLaVA | `fastv` | 3.10 | 2.1.2 | 4.37.2 |
| Qwen2.5-VL / InternVL3 | `vlmevalkit-qwen2.5vl` | 3.11 | 2.7.0+cu126 | 4.48.1 |
| LLaVA-Video | `llavavid` (candidate) | 3.10 | 2.1.2+cu121 | 4.41.2 |

The video source needs a separate Transformers version and should not be installed into the image `fastv` environment. The original video scripts reference `llavavid`; that recovered profile still needs validation. Qwen2.5/InternVL3 uses FlashAttention 2.8.3; video uses 2.6.3. FlashAttention is not included in the recorded image-LLaVA environment.

```bash
git clone git@github.com:MAX-7237/TOPS.git
cd TOPS/VLMEvalKit-TOPS
conda create -n vlmevalkit-qwen2.5vl python=3.11 -y
conda activate vlmevalkit-qwen2.5vl
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-flash-attn.txt
python -m pip check
```

For image LLaVA, create `fastv` with Python 3.10 and install `LLaVA-TOPS/requirements.txt`. For video, create `llavavid` with Python 3.10 and install its own requirements followed by its FlashAttention requirements. Building FlashAttention requires a compatible CUDA toolkit/compiler after PyTorch is installed. Do not bypass resolver failures with `--no-deps`.

The launchers insert their source directories into `PYTHONPATH`; they do not require editable installation. Upstream packaging metadata is retained and may have older constraints. A compatibility guard in the Qwen2.5 TOPS constructor accommodates the older decoder layout and later nested layout, but broader runtime compatibility remains unverified. Candidate dependency repairs mirror MiCo: compatible video Triton/HTTPcore/H11, OpenCV 4.11 for NumPy 1.26, and image PEFT 0.4.0.

## Running

Download weights and datasets separately. Set `HF_HOME` and `LMUData` to your preferred cache/data locations. Models use public Hugging Face IDs by default; VLMEval model locations can be configured in `vlmeval/config.py`. Legacy `/path/to/local-assets` values are placeholders. No weights, datasets, predictions, or infrastructure credentials are included.

Run each command from its component directory. `--dry-run` prints the command without loading weights; remove it to execute.

```bash
# LLaVA-TOPS
python run_tops.py --model llava-1.5-7b --tokens 128 \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava15/answers.jsonl --dry-run
python run_tops.py --model llava-next-13b --tokens 128 \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/next13b/answers.jsonl --dry-run

# VLMEvalKit-TOPS
python run_tops.py --model qwen2.5-vl-7b --tokens 256 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/qwen25 --dry-run
python run_tops.py --model internvl3-8b --tokens 128 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/internvl3 --dry-run

# LLaVA-Video-TOPS
python run_tops.py --model llava-video-7b --tokens 32 --dataset videomme \
  --nproc 1 --work-dir outputs/video --dry-run
```

Image LLaVA's generic entry writes predictions; use the relevant scorer afterwards. Dedicated MMBench and ScienceQA modules remain under `llava/eval`. Video includes the `lmms-eval` adapter, Qwen conversation template, and 64-frame limit; prepare data according to its task configuration. VLMEval supports its registered benchmark adapters, with optional judge-model dependencies for certain tasks.

## Method configuration

The launcher fixes `COVERAGE=SCOPE`, `A1=A2=1`, and both stages' diversity/coverage coefficients to 0.5 for LLaVA and video, or 0.2 for InternVL3. Qwen2.5 defaults to the 0.5/0.5 script variant; another original script used coverage 0.4, so the paper's exact Qwen2.5 result configuration still needs confirmation. These are source-script presets, not a claim of reproduced paper scores.

Image LLaVA sets `STAGE1_KEEP=2T`, multiplied by five for NeXT. The 7B decoder prunes at layers 12 and 24, and 13B at 15 and 30. Qwen2.5 uses layers 12/16 with keep counts 139/39 for T256 and 71/19 for T128. Video uses layers 12/21 with per-frame keep counts 23/7, 11/4, or 5/3 at budgets 64, 32, or 16. The final retained count is therefore different from the nominal layer-average token budget.

## Implementation locations

- Image stage 1: `LLaVA-TOPS/llava/model/llava_arch.py`, branch `prefixvlm_2`.
- Image stage 2: `LLaVA-TOPS/llava/model/language_model/modeling_llama_prefix_2.py`.
- Video: `LLaVA-Video-TOPS/llava/model/llava_arch.py` and `llava/model/language_model/modeling_qwen2_tops.py`.
- Qwen2.5: `VLMEvalKit-TOPS/vlmeval/vlm/qwen2_5_vl/modeling_qwen2_5_vl_tops.py` and its stage-1 dispatch.
- InternVL3: `VLMEvalKit-TOPS/vlmeval/vlm/internvl3/modeling_internvl_chat.py` and `modeling_qwen2_tops.py`.

## Attribution and validation

The implementation accompanies the TOPS manuscript by Tinghao Wang and collaborators supplied for this release. Framework sources build on LLaVA, LLaVA-NeXT/LLaVA-Video, VLMEvalKit, lmms-eval, and Transformers. Existing component licenses and attribution remain in place; consult those licenses before reuse. No new blanket license is assigned to third-party code.

See `VALIDATION.json` for static-check results. Passing syntax and dry-run checks does not establish successful dependency imports, GPU inference, benchmark accuracy, or speedups. Those checks remain pending.
