# T78 — Allow a larger / different label space at test and inference time

status: in-review
tier: 6
type: enhancement
depends_on: T61, T77

## Problem

The label space is fixed at training time: `TrainingPipeline.run` builds the
dense/lexical retrievers and the deployed `LabelSpace` from the classes present
when the model is trained. In production taxonomies grow — new classes appear
after a model ships, and an external test set may reference labels the training
data never contained. Today there is no supported way to:

1. **classify into a class that was added after training**, or
2. **evaluate on a test set whose label space is larger than the training one**,

without retraining. Retraining is not always possible the moment a new class
appears (no examples yet, or the class is genuinely rare), and a growing catalog
is the normal case, not the exception.

The *training-time* half is already supported: a class declared in the
`LabelSpace` with **zero** training examples ships as a description-only class.
What is missing is widening the label space **after** training, at
test/inference time, as a first-class operation.

## Why it's architecturally cheap

The fusion model, calibrator, and abstention policy are already
**label-count-agnostic**: every entry in `FEATURE_NAMES` is a per-candidate
retrieval signal — there is no per-class output dimension anywhere. They score a
`(query, candidate)` feature vector, so adding classes does not change their
input or output shape. Only the retrieval side is indexed by class and needs to
grow. New classes are **appended at the end** so every existing column index
stays stable and the trained model is reused byte-for-byte.

## Known limitation to design around

A class added at inference is **description-only**: `NaN` prototype, no kNN
example support, `class_freq = 0`. The fusion model was trained when every
candidate had example support, so it assigns such a candidate a **low calibrated
confidence**. With precision-tuned abstention thresholds these predictions
typically **abstain / route to human review**. This is honest, not a bug — but
the API and docs must make it explicit.

## Solution (implemented)

1. **`DeployedArtifacts.with_added_classes(new_classes) -> DeployedArtifacts`** —
   returns a new bundle whose label space is extended, reusing
   encoder/fusion/calibrator/abstention and refitting only the retrieval
   description side (dense description embeddings + description BM25). No
   training, no private-attribute surgery — each adapter extends itself
   (`DenseRetrieverAdapter.with_added_classes`,
   `LexicalRetrieverAdapter.with_added_descriptions`).
2. **`InferencePipeline.with_added_classes(new_classes) -> InferencePipeline`** —
   the same operation from the pipeline entry point.
3. **Evaluation with a wider label space**: `text-classifier-eval --classes
   classes.csv` extends the model's label space with any class in the file that
   the model was not trained on (description-only), then scores the labeled set
   against the extended index — so a larger-than-training test set is evaluable
   end-to-end.
4. **Docs + `CHANGELOG.md`**: the description-only confidence caveat and the
   recommended lifecycle (add-at-inference → retrain once `>= n_folds` examples
   exist; the encoder is frozen so retraining is cheap).

Example seeding (giving a new class a few examples for a real prototype/kNN
support without a full retrain) is a documented future extension: the dense side
supports it trivially, but the lexical example BM25 does not retain its corpus,
so a correct symmetric implementation is deferred rather than shipped half-done.

## Acceptance criteria

- [x] A trained model can be extended with new classes at inference with no
      retrain, through a public, documented API (not private-attribute surgery).
- [x] Extended classes are retrievable and can be predicted from their descriptions.
- [x] An external labeled set referencing out-of-training labels can be evaluated
      end-to-end.
- [x] The description-only confidence/abstention behavior is documented and
      covered by tests.
- [x] `CHANGELOG.md` `[Unreleased]` updated.

## Related backlog items

- T77 — user-provided validation/test splits; the wider-label-space eval path
  builds on the same "featurize an external set against the deployment index" idea.
- Zero-example / description-only classes at training time (already supported;
  this ticket is the post-training analogue).
