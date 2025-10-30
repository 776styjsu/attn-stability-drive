# attn-stability-drive (WIP)

## Overview
We investigate whether temporally adjacent driving scenes that look and behave the same also produce stable saliency explanations. The working hypothesis is that, given comparable sensor input and nearly identical control outputs, a well-grounded policy should highlight similar visual evidence each timestep. Large swings in saliency under these conditions suggest that the policy may be relying on fragile or coincidentally correct cues, leaving room for silent failures when circumstances change.

In practical terms, we study saliency-based attributions for learned driving policies and relate saliency stability to notions of input similarity and output agreement. The goal is to flag time ranges where a policy is demonstrably correct but provides inconsistent explanations, so we can triage them for deeper qualitative review and risk assessment.

## Research Questions
- When two frames are near-duplicates in appearance and the policy outputs almost the same control, how consistent are their saliency explanations?
- How frequently do we observe high control agreement paired with low saliency similarity, and do these episodes correlate with risky driving contexts (e.g., occlusions, rapid lighting changes, or sensor noise)?
- Do different similarity metrics (structural, embedding-based, or control-space) agree on which moments indicate unstable explanations?
