"""
services/report_service.py
===========================
Builds the final Markdown market intelligence report.

The report service takes all the evidence gathered by agents and
formats it into a readable, well-structured document.

HALLUCINATION PREVENTION:
- This service only uses data passed to it — it never invents numbers.
- Every statistic in the report has an evidence string with source attribution.
- The "data_limitations" section honestly describes what we don't know.
"""

from __future__ import annotations

from datetime import datetime

from models.agent import MarketReport, ReportSection
from models.resume import ResumeAnalysis
from models.skill import SkillStatistic
from models.trend import TechTrend, TrendSnapshot


class ReportBuilder:
    """
    Fluent builder for MarketReport objects.

    Usage:
        report = (
            ReportBuilder(query="AI Engineer jobs in India")
            .add_skills_section(skill_stats, total_postings=150)
            .add_salary_section(salary_data)
            .add_trends_section(trend_snapshot)
            .add_resume_section(resume_analysis)
            .build()
        )
    """

    def __init__(self, query: str) -> None:
        self._query = query
        self._sections: list[ReportSection] = []
        self._recommendations: list[str] = []
        self._limitations: list[str] = []
        self._sources: set[str] = set()
        self._total_postings = 0
        self._start_time = datetime.utcnow()

    def add_skills_section(
        self,
        skill_stats: list[SkillStatistic],
        total_postings: int,
        country: str | None = None,
    ) -> "ReportBuilder":
        """
        Add a "Top Skills" section showing the most demanded skills.

        Args:
            skill_stats: List of SkillStatistic objects (should be sorted by count).
            total_postings: Total number of postings analyzed.
            country: Optional country filter label.
        """
        self._total_postings = max(self._total_postings, total_postings)
        location_label = f" in {country}" if country else ""

        # Build evidence list from actual data
        evidence = [stat.as_evidence_string() for stat in skill_stats[:20]]

        # Build content table
        rows = ["| Rank | Skill | Postings | Demand |", "|------|-------|----------|--------|"]
        for i, stat in enumerate(skill_stats[:20], 1):
            rows.append(
                f"| {i} | {stat.skill.normalized_name} | "
                f"{stat.occurrence_count}/{total_postings} | "
                f"{stat.percentage}% |"
            )

        content = (
            f"Analysis of **{total_postings} job postings{location_label}** "
            f"from {', '.join(stat.sources[0] for stat in skill_stats[:1])}.\n\n"
            + "\n".join(rows)
        )

        # Collect all sources
        for stat in skill_stats:
            self._sources.update(stat.sources)

        self._sections.append(
            ReportSection(
                title=f"Top Skills{location_label}",
                content=content,
                evidence=evidence,
                sources=list(self._sources),
            )
        )
        return self

    def add_salary_section(
        self,
        role: str,
        min_sal: float | None,
        max_sal: float | None,
        median_sal: float | None,
        currency: str,
        sample_count: int,
        sources: list[str],
        country: str | None = None,
    ) -> "ReportBuilder":
        """
        Add a salary analysis section.

        All values must come from real job data — no invented ranges.
        If sample_count < 5, we warn the user about unreliable data.
        """
        self._sources.update(sources)
        location_label = f" in {country}" if country else ""

        if sample_count < 5:
            self._limitations.append(
                f"Salary data for {role}{location_label} is based on only "
                f"{sample_count} postings — treat as indicative, not definitive."
            )

        if min_sal is None and max_sal is None:
            content = (
                f"Insufficient salary data for {role}{location_label}.\n"
                f"Only {sample_count} postings listed a salary range.\n\n"
                "**This is common** — many employers do not list salaries publicly."
            )
        else:
            content = (
                f"Based on **{sample_count} job postings{location_label}** that listed salary:\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Minimum | {currency} {min_sal:,.0f} |\n"
                f"| Maximum | {currency} {max_sal:,.0f} |\n"
                f"| Median  | {currency} {median_sal:,.0f} |\n\n"
                f"*Note: Only {sample_count} of {self._total_postings} analyzed postings "
                f"disclosed salary information.*"
            )

        evidence = [
            f"Salary data: {sample_count} postings with disclosed ranges — "
            f"Source: {', '.join(sources)}"
        ]

        self._sections.append(
            ReportSection(
                title=f"Salary Analysis for {role}{location_label}",
                content=content,
                evidence=evidence,
                sources=sources,
            )
        )
        return self

    def add_trends_section(
        self, snapshot: TrendSnapshot, top_n: int = 10
    ) -> "ReportBuilder":
        """
        Add a technology trends section.

        Args:
            snapshot: TrendSnapshot with all collected technology data.
            top_n: How many top technologies to show.
        """
        ranked = snapshot.get_ranked(top_n=top_n)

        # Build evidence strings from actual data
        evidence = [t.as_evidence_string() for t in ranked]

        rows = [
            "| Rank | Technology | GitHub Stars | Job Demand | Score |",
            "|------|------------|--------------|------------|-------|",
        ]
        for i, trend in enumerate(ranked, 1):
            stars = (
                f"{trend.github_stats.stars:,}"
                if trend.github_stats
                else "N/A"
            )
            rows.append(
                f"| {i} | {trend.technology} | {stars} | "
                f"{trend.job_demand_percentage}% | {trend.popularity_score:.1f} |"
            )

        content = (
            f"Technology trends based on **{snapshot.query_context}**.\n"
            f"Popularity score combines GitHub activity and job market demand.\n\n"
            + "\n".join(rows)
            + "\n\n*Score formula: GitHub stars (log-scaled, 0–50pts) + "
            "Job demand percentage (0–50pts)*"
        )

        for trend in ranked:
            self._sources.update(trend.sources)

        self._sections.append(
            ReportSection(
                title="Technology Trends",
                content=content,
                evidence=evidence,
                sources=list(self._sources),
            )
        )
        return self

    def add_resume_section(self, analysis: ResumeAnalysis) -> "ReportBuilder":
        """
        Add a resume gap analysis section.

        Args:
            analysis: ResumeAnalysis comparing the user's resume to the market.
        """
        self._sources.update(analysis.sources)

        high_gaps = analysis.get_high_priority_gaps()
        matching_names = [s.normalized_name for s in analysis.matching_skills]
        gap_names = [g.skill.normalized_name for g in high_gaps]

        content = (
            f"**Market Fit Score: {analysis.gap_score:.1f}/100**\n\n"
            f"Based on {analysis.total_postings_analyzed} job postings "
            f"for: *{analysis.query_context}*\n\n"
            f"### ✅ Matching Skills ({len(analysis.matching_skills)})\n"
            f"{', '.join(matching_names) or 'None found'}\n\n"
            f"### ⚠️ High-Priority Gaps ({len(high_gaps)})\n"
            f"These skills appear in >30% of job postings but are missing from your resume:\n\n"
            + (
                "\n".join(
                    f"- **{g.skill.normalized_name}** ({g.market_demand_percentage:.1f}% demand)"
                    for g in high_gaps
                )
                or "None — great coverage!"
            )
        )

        evidence = [
            f"Resume gap analysis: {analysis.total_postings_analyzed} postings analyzed — "
            f"Source: {', '.join(analysis.sources)}"
        ]

        self._sections.append(
            ReportSection(
                title="Resume Gap Analysis",
                content=content,
                evidence=evidence,
                sources=analysis.sources,
            )
        )

        # Auto-add recommendations for high-priority gaps
        for gap in high_gaps[:5]:
            self._recommendations.append(
                f"Learn **{gap.skill.normalized_name}** — "
                f"required in {gap.market_demand_percentage:.1f}% of analyzed job postings."
            )

        return self

    def add_recommendation(self, recommendation: str) -> "ReportBuilder":
        """Add a single recommendation to the report."""
        self._recommendations.append(recommendation)
        return self

    def add_country_comparison_section(self, comparison: dict) -> "ReportBuilder":
        """
        Add a country comparison section.

        Args:
            comparison: Output of the country_comparison_tool.
        """
        countries = comparison.get("countries", {})
        role = comparison.get("role_analyzed", "role")
        sources = comparison.get("sources", ["adzuna"])

        if not countries:
            return self

        self._sources.update(sources)
        rows = [
            "| Metric | " + " | ".join(c for c in countries) + " |",
            "|--------|" + "------|" * len(countries),
        ]

        # Job volume row
        volume = [str(v.get("posting_count", 0)) for v in countries.values()]
        rows.append("| **Postings found** | " + " | ".join(volume) + " |")

        # Salary row
        salary_cells = []
        for v in countries.values():
            sal = v.get("salary", {})
            if sal.get("median"):
                salary_cells.append(
                    f"{sal.get('currency', '')} {sal['median']:,.0f}"
                    f" (n={sal.get('sample_count', 0)})"
                )
            else:
                salary_cells.append("N/A")
        rows.append("| **Median salary** | " + " | ".join(salary_cells) + " |")

        # Top skills row (each country's top 5 in its own column)
        skills_cells = []
        for v in countries.values():
            skills = [s["skill"] for s in v.get("top_skills", [])[:5]]
            skills_cells.append(", ".join(skills) if skills else "N/A")
        rows.append("| **Top skills** | " + " | ".join(skills_cells) + " |")

        insights = comparison.get("cross_country_insights", {})
        insights_block = ""
        common = insights.get("common_skills", [])
        if common:
            insights_block = (
                f"\n\n**Common skills across markets:** {', '.join(common[:8])}"
            )
        unique = insights.get("unique_skills_by_country", {})
        if unique:
            for code, skills in unique.items():
                if skills:
                    insights_block += (
                        f"\n- **Unique to {code}:** {', '.join(skills[:5])}"
                    )

        content = (
            f"Side-by-side job market comparison for **{role}** "
            f"based on live postings from {', '.join(sources)}.\n\n"
            + "\n".join(rows)
            + insights_block
            + "\n\n*Country data reflects the coverage of our job sources, "
            "not the true market size.*"
        )

        evidence = [
            f"Country comparison: {', '.join(countries)} markets compared "
            f"— Source: {', '.join(sources)}"
        ]

        self._sections.append(
            ReportSection(
                title="Country Comparison",
                content=content,
                evidence=evidence,
                sources=sources,
            )
        )
        return self

    def add_limitation(self, limitation: str) -> "ReportBuilder":
        """Add a data limitation note to the report."""
        self._limitations.append(limitation)
        return self

    def build(self, executive_summary: str = "") -> MarketReport:
        """
        Compile all sections into a final MarketReport.

        Args:
            executive_summary: 3–5 sentence overview. If empty, a generic
                               one is generated from the sections.

        Returns:
            A complete MarketReport ready for display.
        """
        end_time = datetime.utcnow()
        execution_time = (end_time - self._start_time).total_seconds()

        # Default limitations for all reports
        default_limitations = [
            "Job posting data may not represent the full market "
            "(only public postings from supported sources are analyzed).",
            "Salary data is only available where explicitly listed in postings "
            "(many companies do not disclose compensation).",
            "Data reflects a point-in-time snapshot and may not capture very recent changes.",
        ]
        all_limitations = default_limitations + self._limitations

        if not executive_summary:
            executive_summary = (
                f"This report analyzes {self._total_postings} job postings "
                f"from {', '.join(sorted(self._sources))} "
                f"in response to the query: '{self._query}'. "
                f"The analysis covers skill demand, technology trends, and "
                f"provides actionable recommendations based on observed market data."
            )

        return MarketReport(
            query=self._query,
            executive_summary=executive_summary,
            sections=self._sections,
            recommendations=self._recommendations,
            data_limitations=all_limitations,
            total_postings_analyzed=self._total_postings,
            sources=sorted(self._sources),
            execution_time_seconds=round(execution_time, 2),
        )
