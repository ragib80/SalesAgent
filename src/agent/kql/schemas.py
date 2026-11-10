"""Schemas for Phase 3 KQL validation results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

IssueSeverity = Literal["error", "warning"]


@dataclass(slots=True)
class KQLValidationIssue:
    """A single validation issue found in generated KQL."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    location: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue representation."""

        return asdict(self)


@dataclass(slots=True)
class KQLValidationResult:
    """Validation outcome for generated or repaired KQL."""

    issues: list[KQLValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[KQLValidationIssue]:
        """Return blocking validation issues."""

        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[KQLValidationIssue]:
        """Return non-blocking validation issues."""

        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """Return whether the KQL is safe enough to execute."""

        return not self.errors

    def add_error(self, code: str, message: str, location: str = "") -> None:
        """Append a blocking issue."""

        self.issues.append(
            KQLValidationIssue(
                code=code,
                message=message,
                severity="error",
                location=location,
            )
        )

    def add_warning(self, code: str, message: str, location: str = "") -> None:
        """Append a non-blocking issue."""

        self.issues.append(
            KQLValidationIssue(
                code=code,
                message=message,
                severity="warning",
                location=location,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable validation summary."""

        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def as_repair_context(self) -> str:
        """Render issues as compact text for a repair prompt."""

        if not self.issues:
            return "No validation issues."
        return "\n".join(
            f"- {issue.severity.upper()} {issue.code}: {issue.message}"
            + (f" ({issue.location})" if issue.location else "")
            for issue in self.issues
        )
