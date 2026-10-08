# Novelty check

Checked 2026-10-08. Sources were read in full and their text is saved in
`paper/citation_checks/` (`cultrace_2508.08879v4.txt`, `chelombitko_2608.02486v1.txt`, PDF
checksums in `novelty_pdfs.sha256`). The search logs are in
`novelty_sweep_arxiv_2026-10-08.json` and `novelty_sweep_anthology_2026-10-08.txt`.

## Verdict

The broad claim "answers or cultures about less-represented groups resolve later in the
network" is **already published**. CulTrace v4 shows it for culture resolution, and
Chelombitko et al. (Aug 2026) show a small version of it for the layer at which the answer
token appears. We cannot present "less-documented means later" as a new finding.

What remains ours is narrower, and it does not overlap with either paper:

- a per-question answer depth, debiased and with its reliability measured;
- within one country, at entity level;
- related to measured pretraining-corpus counts, with length and question controls;
- a preregistered test with a defined null;
- a check of lens depth against patching depth at the level of single questions.

The paper should be pitched as a quantitative, controlled test of that claim. It should not
be pitched as its discovery. This overlap is worse than we expected, and mainly because of
Chelombitko et al., not CulTrace.

## 1. CulTrace v4 (Yu et al., arXiv:2508.08879v4, 21 Aug 2026)

Note on the name: v1 and v2 were titled "Entangled in Representations". Chelombitko et al.
cite this paper as "CultureScope (Yu et al., 2025)", which is where the CultureScope label
in our earlier brief came from. The paper actually titled CultureScope is arXiv:2509.16188,
a different, behavioural benchmark.

### What they measure

| | CulTrace v4 |
|---|---|
| Readout | LatentQA decoders: a LoRA adapter on a decoder LLM, one per layer of the target model (102 adapters). Each adapter is trained to answer natural-language questions about patched-in activations. Decoder output is free text. |
| Readout position | Hidden states of the instruction tokens ("Answer with a single term."), which come after the question, at every layer. The question tokens are deliberately excluded, so that the decoder cannot just read the question (their App. C.2). |
| Metric for "staged reasoning" | Each layer's decoded text ("What is the assistant thinking about?") is labelled by an LLM annotator (Qwen3.6-27B) with a taxonomy: Culture {Unspecified, Misattributed, Proximate, Target}, Domain, Answer {Generated, Candidate}, Reflection, Syntax and Junk. They report the mean layer at which each label appears (their Fig. 4). Annotator accuracy is 96.9% on 100 traces against two humans. |
| Metric for "delayed resolution" | The mean first layer at which the label Culture (Target) appears, per culture (their Table 3 and Table 9). For Llama on BLEnD this runs from 12.8 for China to 21.9 for Algeria, out of 32 layers. A second metric is the accuracy of identifying the target culture, averaged over layers (Table 4). |
| "Less-represented" | Asserted, not measured. Cultures are called high- or low-resourced. There are no corpus counts, pageviews or other frequency measures, and no regression or significance test. The comparison is between 10 culture means. |
| Models | Instruction-tuned: Llama-3.1-8B-Instruct, gemma-3-4b-it, Ministral-8B-Instruct-2410. |
| Data | BLEnD (4,020 short-answer questions) and CANDLE QA (1,916 questions converted from CANDLE assertions by an LLM). Ten countries per dataset, English only. Answers are free-form generation, not multiple choice. |
| Causal validation | None in our sense. "Patching" in CulTrace means inserting activations into the decoder. The decoders are validated by reasoning-recovery similarity, by Gaussian-noise and MMLU controls, and by relational-attribute extraction. There is no activation patching inside the target model. |
| Answer depth | An Answer (Generated) label exists and is placed on the layer axis in Fig. 4, but it is not analysed per culture or against representation. |

### Overlap with us

- Both papers ask about layer-wise processing of cultural knowledge, and both report the
  same direction: knowledge about less-represented cultures is resolved later. Their
  conclusion ("delayed relevant culture resolution ... with less-represented cultures") is
  qualitatively the same as our H1 under its positive outcome.
- Llama-3.1-8B is in both papers, though they use the Instruct model and we use the base
  model.

### Differences (each checked against v4, not assumed)

| Expected difference | Verified? | Notes |
|---|---|---|
| Intra-national vs cross-cultural | **Yes** | CulTrace compares 10 countries. We compare entities across 36 Indian states and union territories, plus a state-level test. |
| Answer stabilisation depth vs culture resolution | **Yes, partly** | Their delay metric is when the decoded text names the target culture. Ours is when the model's answer distribution settles on the gold option. Their Answer (Generated) label is an answer-emergence signal, but it is reported only as a pooled mean layer. |
| Frequency from pretraining-corpus counts | **Yes** | CulTrace has no frequency measure; "less-represented" is a label. We use exact-string counts in Dolma via infini-gram, checked against Wikipedia pageviews (Spearman 0.67), with controls for entity length and question type, and a preregistered test and null. |
| Cyclic-rotation debiasing | **Yes, but not a contrast with CulTrace** | CulTrace uses free-form answers, so letter bias does not arise for them. The debiasing is a contribution for multiple-choice lens studies in general. It is not a way in which CulTrace is wrong. |
| Patching-validated depth | **Yes** | CulTrace has no activation patching in the target model. We patch the entity at every layer and test, question by question, whether patching depth tracks lens depth (H3). Chelombitko et al. do use activation patching (see below), but they aggregate it per model rather than relating it to depth per question. |

Further differences:

- **Readout.** CulTrace uses a trained, generative decoder that is then labelled by an LLM
  judge. We use direct readouts of the model's own answer distribution (logit lens and
  tuned lens), with the last layer checked to reproduce the model's own output logits.
- **Models.** CulTrace uses instruction-tuned models; we use base models.
- **Unit of analysis.** CulTrace compares culture-level means (n = 10). We test per question
  (n ≈ 19,742 questions minus the wrong ones) with controls, so frequency and accuracy can
  be separated.

## 2. Chelombitko et al., "Cultural Awareness is Represented but Not Decoded" (arXiv:2608.02486v1, 3 Aug 2026)

This paper is already cited in the paper draft. On a full reading, it is closer to us than
CulTrace on method.

- **Data and models.** 270 mythological entities (27 Thompson motifs × 10 cultures,
  including Indian) and 18 instruction-tuned models.
- **Instruments:**
  - linear probes for the culture label;
  - a logit lens giving the depth at which the gold entity's first sub-token enters the
    top-k;
  - cross-cultural activation patching;
  - output extraction;
  - a multiple-choice control scored by a restricted first-token log-probability over 3
    option orders.
- **Overlap 1: answer depth by culture.** The logit-lens onset of the gold token is a
  per-culture answer depth. Greek and Roman cross the threshold first, and the other eight
  cultures follow about 1.3 layers later on a 32-layer model (their §4.3, App. I).
- **Overlap 2: patching.** Activation patching locates the causal readout late (median
  peak depth 0.89).
- **Differences:**
  - "less-represented" is again not measured;
  - the comparison is Greco-Roman vs the rest, without a test;
  - the readout is open generation and top-k onset, not a debiased multiple-choice depth;
  - there are no frequency counts or controls;
  - the scale is 270 entities, not about 19.7k questions;
  - the multiple-choice control is not read layer by layer, and letter bias is not
    discussed.

## 3. Sweep for work since August 2026

Searched on 2026-10-08:

- **arXiv API** (search queries are listed in the saved JSON): logit lens or tuned lens
  combined with cultural, regional, India, long-tail, frequency or popularity; SANSKRITI;
  Indian cultural benchmarks combined with mechanistic, interpretability, layers or probing;
  activation patching or causal tracing combined with culture; prediction-depth and
  answer-emergence terms with frequency.
- **ACL Anthology:** the full title lists of the 2026 events published so far (ACL 2026
  with workshops, Findings 2026, EACL 2026, LREC 2026), filtered for culture or India terms
  combined with mechanistic terms, and for lens, long-tail or popularity terms. NAACL,
  COLING and EMNLP 2026 have no event pages yet.
- **Web search** for the same combinations.

Results:

| Work | Date | Relevance | Action |
|---|---|---|---|
| CulTrace v4 (2508.08879v4) | 21 Aug 2026 | High (§1) | Positioned directly in intro and related work |
| Chelombitko et al. (2608.02486v1) | 3 Aug 2026 | High (§2) | Positioning sharpened in intro and related work |
| Floro & Benedetto, "Cultural Binding Heads in Language Models" (2605.28543, v3 9 Sep 2026) | May, revised Sep 2026 | Medium: 2 to 3 mid-layer attention heads causally bind cultural items to identities, in base and instruct models; uses path patching or knockout; no depth or frequency | Cited in related work |
| Huang et al., "The Alignment Paradox" (2609.32617) | 26 Sep 2026 | Low to medium: uses the logit lens on long-tail factual queries; finds that wrong-answer margins grow only in late layers in instruct models; no culture, no frequency-depth link | Cited briefly (logit lens on long-tail facts) |
| Panchal et al., "Indic-TunedLens" (VarDial 2026, ACL Anthology 2026.vardial-1.14) | Mar 2026 (before Aug, found in the sweep) | Medium: a tuned lens for Indian languages on MMLU, from IIT Patna (Ekbal's group); languages, not culture or frequency | Cited in related work |
| Zou et al., "Deciphering Cultural Representations in LLMs via Sparse Autoencoders" (Findings ACL 2026) | Jul 2026 | Medium: culture features found with SAEs, plus ablation and steering; no depth or frequency | Cited in related work |
| Veselovsky et al., "Localized Cultural Knowledge is Conserved and Controllable" (Findings ACL 2026) | Jul 2026 | Medium: an explicit-implicit localisation gap; steering vectors | Cited in related work |
| Zhao et al., "Finding Culture-Sensitive Neurons in VLMs" (EACL 2026) | Mar 2026 | Low: vision-language models; neurons cluster in model-specific layers | Not cited |
| Gurnee et al., "Verbalizable Representations Form a Global Workspace" (2607.15495) | Jul 2026 | Low: argues lens readouts, including the tuned lens, can report what a model is poised to say before it computes it | Noted for Limitations, not cited |
| Anything using SANSKRITI mechanistically | none found | none | none |
| Any other Indian-culture benchmark studied mechanistically | none found (DRISHTIKON 2509.19274 and others are behavioural only) | none | none |

No paper found in the sweep relates measured pretraining frequency to the layer at which
answers settle, and none studies SANSKRITI or another Indian cultural benchmark
mechanistically.

## 4. Implications for the paper

- In the introduction and abstract, do not claim the "later for less-represented"
  direction as new. Claim:
  1. measuring it per entity with corpus counts, controls and a preregistered null;
  2. the within-country setting;
  3. a debiased, reliability-checked multiple-choice depth;
  4. a per-question patching check of lens depth.
- If H1 comes out positive, write it as a confirmation, with measured frequency, of what
  CulTrace and Chelombitko et al. report at culture level.
- If H1 comes out null, that is a substantive disagreement with both papers. It would mean
  their culture-level delays may come from accuracy, question form or culture-level
  confounds rather than from documentation. The paper should then say this carefully,
  since their metrics and readouts differ from ours.
