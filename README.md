<div align="center">

# TOPS

### First-Principles Visual Token Pruning via<br>Constructing Token Optimal Preservation Sets for Efficient MLLM Inference

**Task relevance · Information coverage · Semantic diversity**

[Paper](https://arxiv.org/abs/2606.27161) · [Method](#method) · [Results](#results) · [Quick start](#quick-start) · [Evaluation guide](docs/RUNNING.md)

</div>

---

TOPS constructs a compact visual token set by jointly considering **what matters to the question, what evidence is covered, and what information is complementary**. A training-free, multi-stage pipeline applies these principles before and within the language model.

<table>
<tr>
<th align="center">77.8% visual tokens pruned</th>
<th align="center">100.6% relative performance</th>
<th align="center">7 MLLM backbones</th>
</tr>
<tr>
<td align="center">LLaVA-NeXT · 640-token budget</td>
<td align="center">LLaVA-NeXT-13B · T = 640</td>
<td align="center">Image & video understanding</td>
</tr>
</table>

<a href="assets/teaser-results.pdf"><img src="assets/teaser-results.png" width="100%" alt="Original TOPS performance comparison radar panels from the paper" /></a>

> Figures and performance numbers are from the paper. Current code-release checks are reported separately under [Validation](#validation).

## Method

<a href="assets/method.pdf"><img src="assets/method.png" width="100%" alt="Original TOPS framework and greedy token selection diagram" /></a>

- **Task relevance:** preserve tokens informative for the user query.
- **Information coverage:** represent evidence in the original visual set.
- **Semantic diversity:** favor complementary tokens over redundant ones.

TOPS combines these criteria in greedy subset selection. Stage 1 removes coarse visual redundancy; stage 2 performs text-aware refinement at designated LLM layers.

## Results

### LLaVA model family

<a href="assets/results-table.pdf"><img src="assets/results-table.png" width="100%" alt="Original TOPS LLaVA-series comparison table compiled from LaTeX" /></a>

Relative performance is normalized to each model's unpruned baseline. The original table compares three token budgets across LLaVA-1.5 and LLaVA-NeXT at 7B and 13B scales.

### Inference efficiency

<a href="assets/efficiency-table.pdf"><img src="assets/efficiency-table.png" width="82%" alt="Original TOPS efficiency table on LLaVA-NeXT-7B and POPE" /></a>

The paper reports **4.2T FLOPs versus 41.7T** for the unpruned model at 320 tokens, with POPE F1 of **86.3 versus 86.8**. The table preserves the paper's original latency and memory measurements.

<details>
<summary><strong>Visual evidence retained by TOPS</strong></summary>

<a href="assets/qualitative.pdf"><img src="assets/qualitative.png" width="100%" alt="Original qualitative comparisons of visual token selection" /></a>

</details>

## Supported models

| Model family | Implementation | Environment |
| :--- | :--- | :--- |
| LLaVA-1.5 / LLaVA-NeXT · 7B & 13B | [LLaVA-TOPS](LLaVA-TOPS) | `fastv` |
| LLaVA-Video · 7B | [LLaVA-Video-TOPS](LLaVA-Video-TOPS) | `llavavid` |
| Qwen2.5-VL · 7B / InternVL3 · 8B | [VLMEvalKit-TOPS](VLMEvalKit-TOPS) | `vlmevalkit-qwen2.5vl` |

## Quick start

Use the component-specific requirements in the [environment guide](docs/RUNNING.md#environments):

```bash
git clone https://github.com/MAX-7237/TOPS.git
cd TOPS/VLMEvalKit-TOPS
conda activate vlmevalkit-qwen2.5vl

python run_tops.py --model qwen2.5-vl-7b --tokens 256 \
  --dataset AI2D_TEST --work-dir outputs/tops --dry-run
```

Remove `--dry-run` after preparing dependencies, model weights, and data. See the [full evaluation guide](docs/RUNNING.md#running) for all seven models.

**NeXT budget convention:** pass `128/64/32` to the TOPS launcher. Its five-view scaling corresponds to total budgets `640/320/160`. Video uses per-frame budgets `64/32/16`.

## Validation

This is a **source-preview release**. Python syntax and 19 model/budget launcher dry runs pass. Full environment installation, real-weight GPU inference, and paper-result reproduction remain pending. See [validation records](VALIDATION.json) and the [configuration notes](docs/RUNNING.md#method-configuration).

Figures are copied from the original paper project; tables retain the original LaTeX content and formatting. Click previews for their PDFs. Source paths and hashes are recorded in [asset provenance](assets/SOURCES.json).

## Citation

```bibtex
@article{wang2026tops,
  title={TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference},
  author={Wang, Tinghao and Guo, Yichen and Huang, Rui and others},
  journal={arXiv preprint arXiv:2606.27161},
  year={2026}
}
```

## Acknowledgements

Built on LLaVA, LLaVA-NeXT/LLaVA-Video, VLMEvalKit, lmms-eval, and Transformers. Existing third-party licenses and attribution remain with their components.
