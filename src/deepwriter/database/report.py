from typing import List, Dict, Union
from dataclasses import dataclass

from src.deepwriter.database.document import FlattenBlock


@dataclass
class Section:
    """A completed section of a report.

    This class represents a section of a report, including its title, content,
    original plan, previously written content, and citations.

    Attributes:
        section_title: The title of the section.
        content: The full content of the section.
        plan: The original plan or outline for the section.
        already_written: Content that was already written for this section.
        citations: A list of chunks of information cited in the section.
    """

    section_title: str
    plan: str
    content: str
    already_written: str
    citations: List[FlattenBlock]

    def to_dict(self) -> Dict[str, Union[str, Dict]]:
        return {
            "section_title": self.section_title,
            "plan": self.plan,
            "content": self.content,
            "already_written": self.already_written,
            "citations": [citation.to_dict() for citation in self.citations],
        }


@dataclass
class Report:
    """A complete report composed of multiple sections.

    This class represents a full report, including its sections, formatted content,
    the original query that prompted the report, and vector embeddings.

    Attributes:
        sections: A list of sections that make up the report.
        formatted_report: The complete formatted text of the report.
        query: The original query or prompt that led to the creation of this report.
        embeddings: Vector embeddings representing the semantic content of the report.
    """

    sections: List[Section]
    query: str
    embeddings: List[float]

    def to_dict(self) -> Dict[str, Union[str, List[Dict]]]:
        return {
            "query": self.query,
            "sections": [section.to_dict() for section in self.sections],
        }
