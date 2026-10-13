"""Protocol definitions for leakage-free publication experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 11))
SEEDS = (42, 123, 256)
OBJECTIVES = ("direct", "hybrid", "anchor", "prototype")
CLEAN_ANCHORS = PROJECT_ROOT / "embeddings" / "anchors" / "clean_train_only_uniform_anchors.npz"


@dataclass(frozen=True)
class PublicationRun:
    experiment_id: str
    setting: str
    objective: str
    seed: int
    train_subjects: tuple[str, ...]
    val_subject: str
    test_subject: str

    def to_record(self) -> dict[str, str | int]:
        record = asdict(self)
        record["train_subjects"] = ",".join(self.train_subjects)
        return record


def cross_subject_fold(test_subject: str) -> tuple[tuple[str, ...], str, str]:
    if test_subject not in SUBJECTS:
        raise ValueError(f"Unknown subject: {test_subject}")
    test_index = SUBJECTS.index(test_subject)
    val_subject = SUBJECTS[(test_index + 1) % len(SUBJECTS)]
    train_subjects = tuple(
        subject for subject in SUBJECTS if subject not in {test_subject, val_subject}
    )
    if len(train_subjects) != 8:
        raise AssertionError("Cross-subject fold must contain eight training subjects")
    return train_subjects, val_subject, test_subject


def build_publication_runs(
    *,
    subjects: tuple[str, ...] = SUBJECTS,
    seeds: tuple[int, ...] = SEEDS,
    objectives: tuple[str, ...] = OBJECTIVES,
    include_subject_specific: bool = True,
) -> list[PublicationRun]:
    runs: list[PublicationRun] = []
    for test_subject in subjects:
        train_subjects, val_subject, _ = cross_subject_fold(test_subject)
        subject_tag = test_subject.upper().replace("-", "-")
        for objective in objectives:
            for seed in seeds:
                runs.append(
                    PublicationRun(
                        experiment_id=f"CLEAN-{subject_tag}-{objective.upper()}-S{seed}",
                        setting="cross_subject",
                        objective=objective,
                        seed=seed,
                        train_subjects=train_subjects,
                        val_subject=val_subject,
                        test_subject=test_subject,
                    )
                )
        if include_subject_specific:
            for seed in seeds:
                runs.append(
                    PublicationRun(
                        experiment_id=f"CLEAN-SS-{subject_tag}-DIRECT-S{seed}",
                        setting="subject_specific",
                        objective="direct",
                        seed=seed,
                        train_subjects=(test_subject,),
                        val_subject=test_subject,
                        test_subject=test_subject,
                    )
                )
    experiment_ids = [run.experiment_id for run in runs]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise AssertionError("Publication experiment IDs must be unique")
    return runs
