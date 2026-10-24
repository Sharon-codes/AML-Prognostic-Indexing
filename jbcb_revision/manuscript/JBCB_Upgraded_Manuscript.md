# Stability-guided quantum-behaved feature selection for leukemia transcriptomics: patient-held-out benchmarking, selection reproducibility, and biological source controls

**Sharon Melhi**

Amity School of Engineering and Technology, Amity University Bengaluru, Karnataka, India

**Corresponding author:** Sharon Melhi; sharonmelhi365@gmail.com

## Abstract

**Background and objective:** Sparse transcriptomic classifiers may discriminate leukemia subtypes while yielding poorly reproducible gene panels. We examine whether adding training-set bootstrap recurrence to a quantum-behaved swarm selector improves stability without materially sacrificing held-out discrimination.

**Methods:** We introduced a fixed-size stability-guided binary quantum-behaved particle swarm optimization (SG-BQPSO) objective that combines inner-cross-validation balanced accuracy with bootstrap feature recurrence. We benchmarked eight selectors using three repetitions of five-fold patient-held-out evaluation on Golub leukemia expression data (25 acute myeloid leukemia [AML], 47 acute lymphoblastic leukemia; 7,128 features) and GEO GSE9476 (26 AML, 38 healthy profiles; 22,283 probes). Each classifier used 15 selected features. We quantified area under the receiver operating characteristic curve (AUC), sensitivity, specificity, precision, and Nogueira stability; a curated AML gene list was examined separately for probe coverage and expression differences.

**Results:** SG-BQPSO achieved AUCs of 0.9938 and 0.9936, compared with 0.9844 and 0.9993 for unguided BQPSO. Stability increased from 0.1290 to 0.1830 and from 0.1448 to 0.1683, respectively; several conventional selectors were substantially more stable. SG-BQPSO sensitivity/specificity were 0.9733/0.9433 and 0.8846/0.9912. Of 30 previously curated genes, 23 were represented on GSE9476 and 13 showed exploratory false-discovery-rate-adjusted associations.

**Conclusions:** Bootstrap-anchored swarm search modestly improves selected-set consistency, but accuracy and stability are distinct objectives. Source-aware, held-out evaluation is necessary before biological or clinical claims about sparse signatures.

**Keywords:** acute myeloid leukemia; gene-expression classification; feature-selection stability; quantum-behaved particle swarm optimization; bootstrap consensus; reproducible bioinformatics

## 1. Introduction

Molecular studies of acute myeloid leukemia (AML) have established extensive genetic and phenotypic heterogeneity, motivating computational approaches to discover diagnostic signatures and stratification markers [1,2]. Expression arrays and sequencing platforms measure thousands of transcripts from comparatively small patient cohorts. Sparse classifiers are therefore appealing, but a panel that predicts accurately is not necessarily a panel whose members can be recovered when samples, optimization initializations or assay sources change. Distinguishing discrimination from reproducible molecular discovery is important for biomarker interpretation, assay design and subsequent validation.

Selection instability is especially consequential in high-dimensional data: many correlated or substitutable variables can yield comparable prediction scores, while different training samples or random seeds produce different feature subsets [3,4]. Stability selection uses sample perturbations to emphasize reproducible variables [4], and the chance-adjusted Nogueira measure formalizes the stability of selected sets [3]. In parallel, population-based optimization, including quantum-behaved particle swarm optimization (QPSO), offers flexible search of combinatorial feature spaces [5]. The phrase *quantum-behaved* here denotes a stochastic update equation, not a quantum computer or quantum speedup. Existing approaches do not guarantee that stochastic optimization converges to the same genes merely because its development-set accuracy is high.

A separate threat concerns measurement provenance. Differences in assay platform, specimen preparation, tissue composition and study membership can track outcome labels; classifiers may exploit these differences even when batch correction is applied [6,7]. Reported external AUC values from case-control comparisons are particularly difficult to interpret if all cases originate in one source and all controls in another. A retrospective AML workflow in our repository combined empirical Bayes correction, swarm selection and support-vector classification [8]. Reassessment of its surviving run-level records revealed a mismatch between the high stability originally claimed in manuscript text and the selection lists actually available. This created a practical motivation for a new, fully executed and source-conscious evaluation with explicit subset reproducibility as an independent endpoint.

**Our contribution has three components.** First, we develop a *stability-guided BQPSO* (SG-BQPSO) selection objective, using stratified bootstrap ANOVA recurrence as a training-only reward alongside inner-fold balanced accuracy. Second, we evaluate the method against six nonrandom comparators and a random-feature negative control under identical outer patient splits and a fixed 15-feature budget on two distinct leukemia-expression classification tasks. Third, we connect selection stability to biological follow-up by distinguishing within-cohort discrimination, normal-cell-source sensitivity and exploratory transfer of an earlier curated gene list. This is an incremental computational method with an explicitly testable trade-off, rather than a claim of universal superiority or a clinically validated biomarker panel.

![Figure 1. Evaluation workflow. Every supervised selection and standardization operation is confined to the corresponding outer training fold. SG-BQPSO adds a bootstrap-derived feature-recurrence reward to a fixed-budget swarm search.](figures/Figure_1_Workflow.png)

## 2. Materials and methods

### 2.1. Public cohorts and target definitions

The Golub dataset, originally used to demonstrate molecular leukemia classification [9], provided expression measurements for 72 patients (25 AML, 47 acute lymphoblastic leukemia [ALL]) over 7,128 features. We downloaded the publicly redistributed matrix from the *Computer Age Statistical Inference* data repository. Probe-to-gene correspondence is not preserved in this redistributed transformed matrix; consequently, Golub feature identifiers are treated as numeric indices, and no cross-study gene-level overlap is inferred from them.

GEO accession GSE9476, originally described by Stirewalt and colleagues [10], contains 64 profiles generated with the Affymetrix HG-U133A array (GPL96), including 26 AML samples and 38 samples from healthy hematopoietic cells. Healthy specimens include bone marrow CD34-enriched cells (8), bone marrow mononuclear cells (10), peripheral-blood mononuclear cells (10), and CD34-enriched mobilized peripheral-blood stem cells (10). The downloaded series matrix contains 22,283 probes. The diagnosis field was used to construct labels before model fitting, and sample IDs remained separate across folds. Data checks confirmed 72 and 64 usable profiles, respectively, with no missing probe columns discarded.

The two studies evaluate *different* binary targets (AML versus ALL; AML versus healthy hematopoietic cells) and use different microarray platforms. Their performances must therefore not be interpreted as direct cross-platform transfer or as estimates from the same clinical population. We report them separately throughout. The source URLs, SHA-256 digests and execution parameters appear in Supplementary Table S1 and the executable repository.

### 2.2. Outer evaluation and a common selection budget

For each cohort, we used a repeated stratified five-fold outer split with three repetitions (15 fits per selector), using random seed 20261024. All measurements, model fits and labels of outer test patients were withheld from feature selection and classifier fitting. Within each outer training portion, a one-way ANOVA screen retained the top 80 features; this common pre-screen permitted practical comparison of expensive wrapper methods, but its reliance on class labels makes the selected panel conditional on this first-stage filter. All nonrandom selectors then returned exactly **15 distinct features**. The random-feature negative control sampled 15 features from the full assay space and is interpreted only as a negative control, not a budget-matched optimization baseline.

For the final classifier, expression values for selected features were standardized using means and variances fitted only on the outer training samples, clipped to [-10,10] after scaling, and passed to a class-weighted linear support-vector machine (regularization C=1.0). A fixed decision threshold of zero produced binary predictions; the continuous decision function was used to calculate the receiver operating characteristic area. No test-set threshold optimization was performed. The principal analysis used identical outer splits, classifier family and feature count for all selectors.

### 2.3. Unguided binary quantum-behaved particle swarm optimization

Let \(S\subseteq\{1,\ldots,80\}\) be a feature subset with \(|S|=15\). Each swarm particle stores latent coordinates \(\boldsymbol{x}_i\), a personal-best position \(\boldsymbol{p}_i\) and a shared global best \(\boldsymbol{g}\). The quantum-behaved position update was

\[
\boldsymbol{x}^{(t+1)}_i=\boldsymbol{a}^{(t)}_i \pm \alpha_t\left|\overline{\boldsymbol{p}}^{(t)}-\boldsymbol{x}^{(t)}_i\right|\odot\log(1/\boldsymbol{u}),\qquad
\boldsymbol{a}^{(t)}_i=\boldsymbol{\varphi}\odot\boldsymbol{p}_i+(1-\boldsymbol{\varphi})\odot\boldsymbol{g},
\]

where \(\boldsymbol{u}\) and \(\boldsymbol{\varphi}\) are independent Uniform(0,1) vectors, signs are sampled independently, \(\overline{\boldsymbol{p}}\) denotes the componentwise mean of personal-best positions, and \(\alpha_t\) decreases linearly from 0.95 to 0.50. Latent positions were clipped to [-8,8] and converted to exactly 15 features by ranking coordinates after additive logistic perturbations. This is a reproducible fixed-cardinality BQPSO implementation inspired by, but not identical in hyperparameters to, the historical repository implementation [8]. It does not use quantum hardware.

An unguided particle's fitness was the mean three-fold inner stratified balanced accuracy of a linear support-vector machine evaluated on its selected features. Inner-fold scaling was learned on the corresponding inner training samples. Screening was performed within the *outer* training subset; thus the inner fitness can remain optimistic because the common ANOVA screen uses all outer-training labels, while final outer-test performance is protected against this selection bias [11]. We used ten particles and ten optimization iterations for both BQPSO and conventional binary particle swarm optimization (BPSO).

### 2.4. Bootstrap-anchored stability guidance

SG-BQPSO introduces a second signal that is estimated without accessing outer test samples. Within each outer training portion, six stratified-with-replacement bootstrap samples are drawn. For each bootstrap sample, the 15 highest-ANOVA-ranked genes among the 80 screened candidates are recorded. Feature \(j\)'s recurrence weight is

\[
q_j=\frac{1}{6}\sum_{b=1}^{6}\mathbb{1}\{j\in A_b\},
\]

where \(A_b\) is the bootstrap-specific set. For a particle's proposed 15-feature subset \(S\), the augmented maximization objective is

\[
F_{\lambda}(S)=(1-\lambda)\,\frac{1}{3}\sum_{f=1}^{3}\operatorname{BA}_{f}(S)
+\lambda\,\frac{1}{15}\sum_{j\in S}q_j,
\qquad\lambda=0.15.
\]

Balanced accuracy (BA) guards against class imbalance, while the second term rewards repeated appearance under bootstrap perturbations. The weight 0.15 was specified for the evaluation; it was not optimized against the held-out test outcomes. The unguided BQPSO ablation is \(\lambda=0\), with otherwise matched population size, iteration budget and inner folds. This term is a *soft preference*, not a guarantee of stability, and may trade off discriminative features against repeatability.

### 2.5. Comparator methods

The benchmarks included (i) ANOVA top-15 ranking; (ii) L1-penalized logistic-regression coefficient ranking (C=0.2); (iii) elastic-net logistic-regression coefficient ranking (C=0.3, L1 ratio=0.5); (iv) random-forest importance ranking (150 trees, minimum leaf size two); (v) conventional BPSO with inertia 0.7 and cognitive/social coefficients 1.5; (vi) unguided BQPSO; (vii) SG-BQPSO; and (viii) random 15-feature sampling. Regularized and tree-based selectors operated on the same 80 training-screened features as the swarm methods. Hyperparameters were fixed rather than individually tuned in an extensive algorithm search, so these benchmarks provide controlled evidence about the implemented configurations, not definitive algorithm rankings.

### 2.6. Outcome measures and uncertainty

Within each repetition, predictions from the five held-out folds cover each person once. We calculated AUC, sensitivity, specificity, balanced accuracy and precision for each complete repetition, and present the mean across three repetitions (AUC also with its standard deviation). As repetitions reuse patients, they do **not** create three independent cohorts or independent patient-level observations. Precision is descriptive for the sampled case mix rather than a clinical population-prevalence estimate.

To quantify feature-set consistency across the 15 outer training subsets, we computed the Nogueira index [3], using the full measured feature universe \(d\) and fixed subset size 15:

\[
\widehat{\Phi}=1-\frac{\dfrac{15}{14d}\sum_{j=1}^{d}\widehat{p}_j(1-\widehat{p}_j)}{(15/d)(1-15/d)},
\]

where \(\widehat{p}_j\) is the fraction of outer training folds selecting feature \(j\). We additionally report mean pairwise Jaccard overlap. Both are computed over train-sample perturbations plus any selector stochasticity, and are not estimates from independent populations. A paired stratified patient-bootstrap sensitivity check (4,000 replicates) compared held-out AUC and balanced accuracy between SG-BQPSO and unguided BQPSO using **one complete outer repetition**; the resulting percentile intervals are conditional on the fitted fold models, not resampling-adjusted confidence intervals for a complete retraining pipeline.

### 2.7. Exploratory biological and provenance controls

The original development repository [8] also includes an independently curated 30-gene signature. As an *exploratory biological transfer check*, we mapped that list to the official GPL96 gene-symbol annotation; a gene with multiple probes was represented by the within-patient median of its probes. We compared AML and healthy groups using Welch's t-test, applying Benjamini–Hochberg correction among the **23 represented genes**. These tests do not estimate the performance of the original frozen AML classifier and do not adjust for shifts in hematopoietic cell composition.

A separate source-audit reconstructed feature-selection frequencies from 30 archived outputs (30 features each) and examined control-source-specific confusion summaries. The archived selections were generated before the later gene-panel remediation and do not prove stability of that curated panel. Where original cohort-level records involved TCGA RNA-seq cases and microarray-only controls, platform and disease are inseparable by design, precluding an unconfounded diagnostic interpretation. Such records are considered motivation and a provenance control, not the new benchmark outcomes.

### 2.8. Computational reproducibility

The full Python implementation, exact public data URLs, input SHA-256 digests, seeds, selection counts, outer-fold prediction tables, feature-frequency lists and GitHub Actions execution logs are provided on the versioned branch at **https://github.com/Sharon-codes/AML-Prognostic-Indexing/tree/jbcb-evidence-revision-20261024**. The benchmark job downloads expression matrices into a temporary workspace and commits only derived analysis outputs. All results in Sections 3.1–3.4 arise from the newly executed benchmark, whereas the retrospective audit in Section 3.5 is labelled separately. Supplementary Information specifies command-line instructions and the statistical scope of each result.

## 3. Results

### 3.1. Two real-expression benchmarks completed with identical outer-fold designs

All eight selectors completed 15 patient-held-out outer evaluations on both datasets. No test patients were included in the training-only 80-candidate screen or final classifier standardization. The Golub analysis involved 25 AML and 47 ALL patients, while GSE9476 involved 26 AML and 38 healthy hematopoietic profiles, including four normal specimen subtypes. The primary held-out performance and selected-set stability statistics appear in Table 1 and Figures 2–4.

**Table 1.** Mean ROC AUC (standard deviation across three complete repetitions) and Nogueira stability across 15 outer selection sets. Each row uses 15 features, a common outer split and linear SVM final classifier. These are within-study held-out results, not independent external-site test results.

| Selection method | Golub AUC ± SD | Golub stability | GSE9476 AUC ± SD | GSE9476 stability |
|:---|---:|---:|---:|---:|
| ANOVA | 0.9929 ± 0.0038 | **0.7442** | 0.9939 ± 0.0088 | **0.5896** |
| L1 logistic | 0.9923 ± 0.0045 | 0.6392 | 0.9966 ± 0.0021 | 0.4898 |
| Elastic net | 0.9921 ± 0.0068 | 0.5101 | 0.9939 ± 0.0088 | 0.4009 |
| Random forest | **0.9946 ± 0.0018** | 0.6399 | 0.9882 ± 0.0015 | 0.3691 |
| BPSO | 0.9850 ± 0.0077 | 0.1360 | **0.9993 ± 0.0012** | 0.1423 |
| Unguided BQPSO | 0.9844 ± 0.0055 | 0.1290 | **0.9993 ± 0.0012** | 0.1448 |
| **SG-BQPSO (proposed)** | 0.9938 ± 0.0018 | **0.1830** | 0.9936 ± 0.0021 | **0.1683** |
| Random 15 (negative control) | 0.7989 ± 0.0561 | −0.0015 | 0.7659 ± 0.0481 | ≈0 |

![Figure 2. Held-out ROC AUC across all eight selectors. Error bars show the standard deviation across three full cross-validation repetitions; they are not confidence intervals from independent cohorts.](figures/Figure_2_Classification.png)

![Figure 3. Nogueira stability across 15 outer training subsets. The stability-guided swarm improved on unguided BQPSO, while simpler selection approaches were substantially more consistent.](figures/Figure_3_Stability.png)

### 3.2. The bootstrap reward improves BQPSO stability, not every performance measure

On Golub, adding bootstrap guidance increased Nogueira stability from **0.1290** to **0.1830** (an absolute increase of 0.0541), and mean AUC from 0.9844 to 0.9938. On GSE9476, stability increased from **0.1448** to **0.1683** (0.0235 absolute), whereas mean AUC decreased from 0.9993 to 0.9936. Pairwise Jaccard likewise increased from 0.0720 to 0.1042 in Golub and from 0.0800 to 0.0943 in GSE9476. These observations establish a modest but consistent *within-swarm* stability effect of the chosen guidance term under the tested budgets; they do not demonstrate global superiority.

Most importantly, the ANOVA and penalized models were appreciably more stable under the same panel size and splits. For instance, ANOVA reached Nogueira indices of 0.7442 and 0.5896 in the two datasets, compared with 0.1830 and 0.1683 for SG-BQPSO. Thus, a higher-dimensional swarm search should not be justified merely by an accuracy score near one: simple feature selectors remain essential controls in any claim of reproducible biomarker discovery. Random subsets yielded lower discrimination (mean AUC 0.7989 and 0.7659), but even these values were noticeably above 0.5 on these small expression datasets.

![Figure 4. Discrimination–stability trade-off for the eight methods. The additional consistency introduced by SG-BQPSO is relative to the unguided swarm, not relative to ANOVA or penalized selection.](figures/Figure_4_Pareto.png)

### 3.3. Patient-level sensitivity, specificity and conditional paired contrasts

SG-BQPSO sensitivity/specificity across the three complete held-out passes were 0.9733/0.9433 for Golub and 0.8846/0.9912 for GSE9476; its corresponding balanced accuracies were 0.9583 and 0.9379. Pooling each patient's one prediction per repetition for a purely descriptive precision calculation gives 73 true-positive and 8 false-positive predictions out of 216 held-out predictions for Golub (precision 0.9012); in GSE9476 the corresponding counts are 69 true positives and 1 false positive out of 192 held-out predictions (precision 0.9857). The three representations of each patient are correlated and must not be treated as independent clinical observations.

The first full outer repetition was additionally subjected to a paired, class-stratified patient bootstrap. The SG–BQPSO AUC difference was +0.0077 (conditional 95% percentile interval −0.0034 to +0.0247) in Golub and −0.0020 (−0.0121 to +0.0020) in GSE9476. Both intervals include zero. Therefore we do not claim a statistically demonstrated gain in AUC. The corresponding first-repetition balanced-accuracy difference was +0.0506 (interval 0.0000 to 0.1119) in Golub and exactly zero for GSE9476. These intervals condition on the already selected and fitted fold models; they do not include full fitting-process variation.

**Table 2.** Descriptive SG-BQPSO performance, averaged over three complete outer repetitions. Precision is pooled across the three correlated out-of-fold passes only to disclose classification counts.

| Outcome | Golub (AML vs ALL) | GSE9476 (AML vs healthy) |
|:---|---:|---:|
| Mean AUC | 0.9938 | 0.9936 |
| Mean sensitivity | 0.9733 | 0.8846 |
| Mean specificity | 0.9433 | 0.9912 |
| Mean balanced accuracy | 0.9583 | 0.9379 |
| Descriptive pooled precision | 0.9012 | 0.9857 |
| Nogueira stability | 0.1830 | 0.1683 |

### 3.4. Control-cell subtypes remain an important biological evaluation dimension

GSE9476 contained healthy controls from four distinct specimen categories rather than a single matched healthy state. The unguided BQPSO method classified all evaluated control predictions correctly across the three repeated held-out passes; SG-BQPSO produced one false-positive prediction among 24 correlated predictions representing eight bone-marrow CD34-enriched profiles, and none among the other control categories. This is a useful subgroup descriptive analysis, but it neither resolves age/specimen differences nor provides an independent external validation study. The source label itself is close to the disease label in the GEO metadata and cannot be used as an independent biological ground truth.

### 3.5. Exploratory follow-up of the earlier 30-gene panel

Twenty-three of the previously curated 30 gene identifiers were represented by at least one GPL96 probe using the official GEO annotation; seven had no exact annotation match. Thirteen of these 23 displayed a Benjamini–Hochberg-adjusted \(q<0.05\) in a descriptive AML-versus-healthy Welch contrast. Examples include **CDKN2A** (\(q=1.46\times10^{-8}\)), **RTN1** (\(q=4.50\times10^{-5}\)), **TPM1** (\(q=4.74\times10^{-5}\)), and **XCL1** (\(q=0.00296\)). In the same cohort, **OAZ1**, **TREM1** and **PRPF8** did not show adjusted evidence of differential expression (\(q=0.527\), 0.682 and 0.682). These are exploratory transcript-level associations in a single case–control study and do not establish clinical utility or lack thereof. The full probe-coverage and multiplicity-adjusted table is supplied as Supplementary Table S4.

![Figure 5. Exploratory GPL96 expression associations for mapped genes in the prior curated panel. Results use within-patient median of mapped probes and false-discovery-rate adjustment over the 23 assayed genes; direction reflects AML minus control expression.](figures/Figure_5_Exploratory_Genes.png)

### 3.6. A provenance audit explains the need for the revised validation strategy

A separate reconstruction of 30 previously archived swarm outputs showed 900 total selected-feature instances spanning 551 distinct gene identifiers, with 333 genes appearing only once and the maximum recurrence of any gene being 15/30. The chance-adjusted Nogueira index was approximately **0.0204** with a 1,000-feature candidate denominator; no gene met an 80% recurrence criterion. A newly executed 10,000-replicate uniform-selection reference benchmark yielded an empirical upper-tail Monte Carlo estimate \(p\approx 0.00010\), demonstrating greater-than-uniform recurrence but *not* high absolute stability. The archived 30 runs predate the later curated panel and cannot validate that panel's reported run stability.

Retrospective control-source summaries likewise differed: one record reported specificity 51/51 for array-based bulk controls, while another reported 30/50 for a distinct control source, despite similar AML sensitivity. When TCGA RNA-seq provides all external AML cases and a separate microarray source provides all controls, source and diagnosis are inseparable within the test design. These historical numbers are not substituted for the new patient-held-out results and are included as an audit rather than an additional newly fitted classifier experiment.

## 4. Discussion

### 4.1. What the incremental algorithm contributes

The method evaluated here adds a mathematically explicit, training-only preference for features that recur under bootstrap perturbations. Unlike a post hoc claim that the final swarm-selected panel is stable, this approach incorporates recurrence into the search objective itself, and evaluates both the objective's predictive trade-off and realized selected-set stability under patient-held-out splits. The same fixed 15-feature budget, common screened universe, linear classifier and outer folds allow a meaningful within-study comparison with unguided BQPSO. On two leukemia-expression tasks, the observed stability increase was consistent in direction but modest in magnitude; it is better framed as a controllable trade-off than as a step change in biomarker reproducibility.

The conventional ANOVA, L1 and elastic-net selectors were much more stable, while remaining highly discriminative. This is not a weakness to conceal: it clarifies when sophisticated stochastic wrappers are unnecessary. Under low sample sizes and strongly correlated expression features, many sparse subsets can retain class-separating information; optimizing a classification margin does not ensure that a particular mechanistic gene is consistently selected. Further work should examine whether richer fitness functions, alternative stability penalties or constrained pathway-level selections can improve the Pareto frontier relative to simple filters without introducing overfitting.

### 4.2. Why cohort and endpoint specificity matters

Golub distinguishes AML from another leukemia subtype, whereas GSE9476 distinguishes AML profiles from diverse normal hematopoietic specimens. These are different medical classification questions, with different assay platforms and source distributions. The near-perfect held-out AUCs reported for several methods on these small studies should be interpreted as *within-study classification*, not readiness for clinical diagnosis. In particular, cells sampled from marrow and blood differ in composition, and patient disease may correlate with processing or isolation protocols. The analysis of GSE9476 healthy subgroups suggests the importance of reporting source-specific false positives even when aggregate specificity is high.

The historical source audit illustrates a stronger version of this concern. An external design with all cases on RNA sequencing and all controls on microarrays cannot by itself prove disease-invariant cross-platform discrimination, irrespective of the batch-harmonization tool. Empirical Bayes correction can help adjust batch effects [6,7], but does not create missing case/control combinations or guarantee that technical variation has been removed without damaging biology. This motivates future balanced external cohorts with both labels represented on each platform, fully locked preprocessing and documented patient relationships across studies.

### 4.3. Interpreting the biological follow-up without claiming prognostic validation

The exploratory mapping of the earlier 30-gene list to GSE9476 was intended to test annotation coverage and association, not to imply that the original multi-platform classifier transferred to this dataset. Thirteen mapped genes passed correction within a deliberately limited 23-gene testing family, whereas three previously emphasized genes (OAZ1, TREM1 and PRPF8) did not. These observations should guide independent validation rather than be turned into mechanistic statements: transcript abundance, RNA processing, cell mixture, protein interactions and clinical outcomes are different biological measurements. The newer study does not establish a survival-risk model, Kaplan–Meier separation, drug target or pathway mechanism. Claims of prognosis would require individual observed survival times, censoring information and independent risk-score validation.

The GPL96 annotation also introduces probe ambiguity and historical gene names, and the original Golub transformed matrix lacks preserved gene identifiers. A meaningful cross-cohort biological replication study should therefore use uniformly mapped modern expression units, pre-specified gene identifier handling, sample-level covariates and independent donor cohorts. The 23-gene expression contrast is supportive only of further hypothesis generation.

### 4.4. Limitations, reproducibility and appropriate next studies

The sample sizes (72 and 64) limit precision, especially for subgroup specificity, and the repeated outer evaluations reuse patients. The bootstrapped AUC-difference intervals condition on fixed fitted models from a single repetition, not all sources of model-training uncertainty. Only a single stability coefficient (0.15), fixed subset size (15), candidate size (80), and finite ten-particle/ten-iteration search budget were studied; these are computationally tractable choices rather than optimized prescriptions. Common ANOVA screening, although restricted to each outer training fold, makes all methods dependent on one preliminary filter and gives a potentially optimistic internal fitness objective. Bootstrap anchor scores partly inherit the preferences of ANOVA, so greater stability should not be mistaken for recovery of independent biological truth.

Neither study provides a truly independent external institution/time-period test of a model frozen before validation, and a fully controlled cross-platform comparison could not be constructed from these two different endpoints. Technical sources and tissue compartments are imperfectly matched. The study also lacks experimentally verified proteins or functions. Nevertheless, the complete selection, prediction and recurrence records are released, so the observed stability–discrimination patterns can be recalculated without relying on hand-drawn figures or untraceable headline statistics.

Future work should pre-register the stability–accuracy objective, evaluate more values of \(\lambda\) and selection budgets under matched evaluation counts, implement site- or donor-group-held-out validation on balanced multi-platform AML/control data, and compare optimized conventional selectors and pathway-based penalties. Such experiments would determine whether SG-BQPSO provides added utility beyond ANOVA and elastic-net approaches, rather than assuming that the swarm architecture is intrinsically advantageous.

## 5. Conclusions

This study presents and tests a bootstrap-anchored stability reward for fixed-cardinality quantum-behaved swarm feature selection. Across two publicly accessible leukemia-expression cohorts, the added reward modestly increased feature-selection stability relative to an unguided swarm while preserving high, but not uniformly higher, held-out AUC. Simpler statistical selectors remained more stable. Independent phenotype definitions, source-aware subgroup analysis and exploratory probe-annotation checks highlight why discriminative accuracy, consistent feature recovery and biological relevance must be evaluated separately. The executable benchmark provides a reproducible platform for the next generation of leakage-controlled, source-balanced transcriptomic biomarker studies.

## Acknowledgements and declarations

**Acknowledgements.** The author acknowledges the investigators and patients represented in the Golub leukemia study and GEO GSE9476, the Gene Expression Omnibus and the original AML genomic research consortia.

**Ethics approval.** This study reanalysed publicly released, de-identified transcriptomic expression profiles and computational outputs. No new human participants were recruited or samples collected; ethical oversight for primary collection belongs to the original studies and their applicable governance conditions.

**Funding.** No external funding was reported for this computational work.

**Competing interests.** The author declares no known competing financial or personal interests.

**Data and code availability.** Reproducible scripts, outer-fold predictions, study metadata, result tables and execution logs are available at https://github.com/Sharon-codes/AML-Prognostic-Indexing/tree/jbcb-evidence-revision-20261024. Original public matrices are sourced from the documented Golub and GEO URLs; raw patient-level expression matrices are not republished in the repository. A submission-specific ZIP contains the supporting scripts, figures and provenance checks.

**Author contributions.** Sharon Melhi: conceptualization, methodology, software development, experiments, analysis, interpretation, manuscript preparation and critical revision. The author is responsible for checking and approving the submitted manuscript and all findings.

**Declaration of assisted tools.** Automated computational and language tools assisted with executable analysis, code review and manuscript preparation. The author is responsible for independent verification of the final scientific content, primary data provenance and journal disclosures.

## References

[1] H. Döhner, A. H. Wei, F. R. Appelbaum, et al., “Diagnosis and management of AML in adults: 2022 recommendations from an international expert panel on behalf of the ELN,” *Blood* **140** (12), 1345–1377 (2022). doi:10.1182/blood.2022016867.

[2] Cancer Genome Atlas Research Network, “Genomic and epigenomic landscapes of adult de novo acute myeloid leukemia,” *New England Journal of Medicine* **368**, 2059–2074 (2013). doi:10.1056/NEJMoa1301689.

[3] S. Nogueira, K. Sechidis and G. Brown, “On the stability of feature selection algorithms,” *Journal of Machine Learning Research* **18**(174), 1–54 (2018). https://jmlr.org/papers/v18/17-514.html.

[4] N. Meinshausen and P. Bühlmann, “Stability selection,” *Journal of the Royal Statistical Society: Series B* **72**(4), 417–473 (2010). doi:10.1111/j.1467-9868.2010.00740.x.

[5] J. Sun, B. Feng and W. Xu, “Particle swarm optimization with particles having quantum behavior,” in *Proceedings of the IEEE Congress on Evolutionary Computation*, 325–331 (2004). doi:10.1109/CEC.2004.1330875.

[6] W. E. Johnson, C. Li and A. Rabinovic, “Adjusting batch effects in microarray expression data using empirical Bayes methods,” *Biostatistics* **8**(1), 118–127 (2007). doi:10.1093/biostatistics/kxj037.

[7] J. T. Leek, R. B. Scharpf, H. C. Bravo, et al., “Tackling the widespread and critical impact of batch effects in high-throughput data,” *Nature Reviews Genetics* **11**, 733–739 (2010). doi:10.1038/nrg2825.

[8] S. Melhi, *AML-Prognostic-Indexing: Public computational workflow and historical run evidence* (versioned source repository, accessed 24 October 2026). https://github.com/Sharon-codes/AML-Prognostic-Indexing.

[9] T. R. Golub, D. K. Slonim, P. Tamayo, et al., “Molecular classification of cancer: Class discovery and class prediction by gene expression monitoring,” *Science* **286**(5439), 531–537 (1999). doi:10.1126/science.286.5439.531.

[10] D. L. Stirewalt, S. Meshinchi, K. J. Kopecky, et al., “Identification of genes with abnormal expression changes in acute myeloid leukemia,” *Genes, Chromosomes & Cancer* **47**(1), 8–20 (2008). doi:10.1002/gcc.20500.

[11] C. Ambroise and G. J. McLachlan, “Selection bias in gene extraction on the basis of microarray gene-expression data,” *Proceedings of the National Academy of Sciences USA* **99**(10), 6562–6566 (2002). doi:10.1073/pnas.102102699.

[12] G. S. Collins, K. G. M. Moons, P. Dhiman, et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” *BMJ* **385**, e078378 (2024). doi:10.1136/bmj-2023-078378.

[13] S. Warnat-Herresthal, H. Schultze, K. L. Shastry, et al., “Swarm Learning for decentralized and confidential clinical machine learning,” *Nature* **594**, 265–270 (2021). doi:10.1038/s41586-021-03583-3.