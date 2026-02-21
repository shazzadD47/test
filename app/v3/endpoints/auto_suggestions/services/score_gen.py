from app.v3.endpoints.auto_suggestions.configs import settings
from app.v3.endpoints.auto_suggestions.schemas import (
    ImageSelection,
    ImageSuggested,
    Keyword,
)


class ConfidenceScoreGenerator:
    def __init__(
        self,
        keywords_extraction_info: dict,
        auto_suggestion_llm_output: list[ImageSelection],
    ):
        self.keywords_extraction_info: dict = keywords_extraction_info
        self.auto_suggestion_llm_output: list[ImageSelection] = (
            auto_suggestion_llm_output
        )
        self._add_mandatory_keyword_match_count()
        self.mandatory_contribution = settings.MANDATORY_KEYWORD_CONTRIBUTION
        self.optional_contribution = 0.99 - self.mandatory_contribution

    def _add_mandatory_keyword_match_count(self) -> list[dict]:
        mandatory_keywords = self.keywords_extraction_info.get("mandatory_keywords")
        if mandatory_keywords:
            for output in self.auto_suggestion_llm_output:
                keyword_matches = output.get("keyword_matches", [])
                mandatory_match = [
                    kw for kw in keyword_matches if kw in mandatory_keywords
                ]
                output["mandatory_keyword_match_count"] = len(mandatory_match)

    def _get_independent_optional_keywords(
        self, sorted_keywords: list[Keyword], optional_keywords: list[str]
    ) -> set[str]:
        final_independent_optional_keywords = set()
        excluded = set()
        for item in sorted_keywords:
            keyword = item["keyword"]
            if keyword in optional_keywords and keyword not in excluded:
                final_independent_optional_keywords.add(keyword)
                excluded.update(item["fallback_keywords"])
        return final_independent_optional_keywords

    def _get_contribution_mapping(
        self,
        sorted_keywords: list[Keyword],
        final_independent_optional_keywords: set[str],
    ) -> list[dict]:
        contribution_mapping = []
        for independent_optional_keywords in final_independent_optional_keywords:
            contribution_mapping.append(
                {
                    "keyword": independent_optional_keywords,
                    "contribution": self.optional_contribution
                    / len(final_independent_optional_keywords),
                }
            )

        for item in sorted_keywords:
            keyword = item["keyword"]
            if keyword in final_independent_optional_keywords:
                for mapping in contribution_mapping:
                    if mapping["keyword"] == keyword:
                        confidence = mapping.get("contribution")
                        break
                if confidence:
                    fallback_keywords = item["fallback_keywords"]
                    for item in sorted_keywords:
                        keyword_2 = item["keyword"]
                        if keyword_2 in fallback_keywords:
                            contribution_mapping.append(
                                {
                                    "keyword": keyword_2,
                                    "contribution": confidence / item["priority"],
                                }
                            )

        return contribution_mapping

    def _get_suggested_image_data(
        self,
        mandatory_keywords_count: int,
        optional_keywords: list[str],
        contribution_mapping: list[dict],
    ) -> list[ImageSuggested]:
        suggested_image_data = []
        for data in self.auto_suggestion_llm_output:
            if not data["keyword_matches"]:
                continue
            data_mandatory_keyword_count = data["mandatory_keyword_match_count"]
            data["confidence_score"] = (
                self.mandatory_contribution
                * data_mandatory_keyword_count
                / mandatory_keywords_count
            )
            for kw_optional_check in data["keyword_matches"]:
                if kw_optional_check in optional_keywords:
                    for mapping in contribution_mapping:
                        if mapping["keyword"] == kw_optional_check:
                            confidence = mapping.get("contribution")
                            data["confidence_score"] = (
                                data["confidence_score"] + confidence
                            )
            suggested_image_data.append(data)
        return suggested_image_data

    def get_confidence_score_output(self) -> list[ImageSuggested]:
        mandatory_keywords: list[str] = self.keywords_extraction_info[
            "mandatory_keywords"
        ]
        mandatory_keywords_count = len(mandatory_keywords)

        optional_keywords: list[str] = self.keywords_extraction_info[
            "optional_keywords"
        ]
        optional_keywords_info: list[Keyword] = self.keywords_extraction_info[
            "optional_keywords_info"
        ]

        # Detect independent optional keyword
        sorted_keywords = sorted(optional_keywords_info, key=lambda x: x["priority"])
        final_independent_optional_keywords = self._get_independent_optional_keywords(
            sorted_keywords, optional_keywords
        )
        contribution_mapping = self._get_contribution_mapping(
            sorted_keywords, final_independent_optional_keywords
        )
        suggested_image_data = self._get_suggested_image_data(
            mandatory_keywords_count, optional_keywords, contribution_mapping
        )
        return suggested_image_data
