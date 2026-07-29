#!/usr/bin/env python3
"""
Project dynamics statistics from the GitHub repository mining dataset.

The dataset (results.csv.gz) has one row PER REPOSITORY (a snapshot), not
one row per issue. There are no per-issue timestamps, only:
  - createdAt / pushedAt / updatedAt (repo-level dates)
  - totalIssues / openIssues (cumulative counts at scrape time)

So "issues reported per month" cannot be read off directly as a real
calendar time series. Instead this script estimates, per project, an
average issue-reporting rate:

    issues_per_month = totalIssues / age_in_months

where age_in_months is the time between repo creation and its last
recorded update (updatedAt, falling back to pushedAt). Aggregating this
rate by the repo's creation-month cohort gives a trend of how issue
reporting intensity has evolved across the dataset over calendar time,
which is the closest available proxy to a real monthly issue trend.

By default the analysis is restricted to the 10000 projects with the most
stargazers (use --top-stars 0 to analyse all projects instead), and further
restricted to projects that average at least 10 commits/week over their
lifetime (use --min-commits-per-week 0 to disable), as a proxy for
"consistently active" projects. The star-based restriction is applied
among the projects that pass the activity filter, i.e. the top N by stars
are selected from the already-filtered, active subset.

Usage:
    python3 project_dynamics_stats.py [--input results.csv.gz]
                                       [--output-dir stats]
                                       [--top-stars 10000]
                                       [--min-commits-per-week 10]
                                       [--plot]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.44

USE_COLS = [
    "name",
    "isFork",
    "isArchived",
    "mainLanguage",
    "createdAt",
    "pushedAt",
    "updatedAt",
    "totalIssues",
    "openIssues",
    "totalPullRequests",
    "openPullRequests",
    "stargazers",
    "forks",
    "contributors",
    "commits",
]


def load_data(
    input_path: Path,
    top_stars: int | None = None,
    min_commits_per_week: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=USE_COLS)
    for col in ("createdAt", "pushedAt", "updatedAt"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["endDate"] = df["updatedAt"].fillna(df["pushedAt"])
    age_days = (df["endDate"] - df["createdAt"]).dt.days
    df["ageMonths"] = (age_days.clip(lower=1) / DAYS_PER_MONTH)
    df["ageWeeks"] = (age_days.clip(lower=1) / 7)

    valid = df["createdAt"].notna() & df["endDate"].notna() & (age_days > 0)
    df = df[valid].copy()

    df["commitsPerWeek"] = df["commits"] / df["ageWeeks"]
    if min_commits_per_week is not None:
        df = df[df["commitsPerWeek"] >= min_commits_per_week].copy()

    if top_stars is not None:
        df = df.nlargest(top_stars, "stargazers").copy()

    df["issuesPerMonth"] = df["totalIssues"] / df["ageMonths"]
    df["pullRequestsPerMonth"] = df["totalPullRequests"] / df["ageMonths"]
    df["closedIssues"] = df["totalIssues"] - df["openIssues"]
    df["issueCloseRate"] = np.where(
        df["totalIssues"] > 0, df["closedIssues"] / df["totalIssues"], np.nan
    )
    df["creationCohort"] = df["createdAt"].dt.to_period("M")
    return df


def summary_overall(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "totalIssues",
        "openIssues",
        "issuesPerMonth",
        "pullRequestsPerMonth",
        "issueCloseRate",
        "commitsPerWeek",
        "ageMonths",
    ]
    groups = {
        "all_projects": df,
        "non_forks": df[~df["isFork"]],
        "non_forks_non_archived": df[~df["isFork"] & ~df["isArchived"]],
    }
    rows = []
    for label, subset in groups.items():
        desc = subset[metrics].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
        desc.insert(0, "group", label)
        desc.index.name = "stat"
        rows.append(desc.reset_index())
    return pd.concat(rows, ignore_index=True)


def issues_per_month_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    groups = {
        "all_projects": df,
        "non_forks": df[~df["isFork"]],
        "non_forks_non_archived": df[~df["isFork"] & ~df["isArchived"]],
    }
    rows = []
    for label, subset in groups.items():
        values = subset["issuesPerMonth"].dropna()
        row = {"group": label, "count": len(values)}
        for p in percentiles:
            row[f"p{p}"] = np.percentile(values, p) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def issues_per_month_by_cohort(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[~df["isFork"]]
    grouped = subset.groupby("creationCohort").agg(
        numProjects=("name", "count"),
        totalIssuesReported=("totalIssues", "sum"),
        meanIssuesPerMonth=("issuesPerMonth", "mean"),
        medianIssuesPerMonth=("issuesPerMonth", "median"),
        meanAgeMonths=("ageMonths", "mean"),
    )
    grouped = grouped.sort_index()
    grouped.index = grouped.index.astype(str)
    return grouped.reset_index().rename(columns={"creationCohort": "cohortMonth"})


def stats_by_language(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    subset = df[~df["isFork"]]
    top_languages = subset["mainLanguage"].value_counts().head(top_n).index
    filtered = subset[subset["mainLanguage"].isin(top_languages)]
    grouped = filtered.groupby("mainLanguage").agg(
        numProjects=("name", "count"),
        meanIssuesPerMonth=("issuesPerMonth", "mean"),
        medianIssuesPerMonth=("issuesPerMonth", "median"),
        meanIssueCloseRate=("issueCloseRate", "mean"),
        meanStargazers=("stargazers", "mean"),
    )
    return grouped.sort_values("numProjects", ascending=False).reset_index()


def activity_correlations(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[~df["isFork"]]
    cols = [
        "issuesPerMonth",
        "stargazers",
        "forks",
        "contributors",
        "commits",
        "pullRequestsPerMonth",
    ]
    return subset[cols].corr(method="spearman")


def make_plots(df: pd.DataFrame, cohort_stats: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = df[~df["isFork"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    finite = subset["issuesPerMonth"].replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite[finite <= finite.quantile(0.99)]
    ax.hist(finite, bins=60)
    ax.set_xlabel("Issues reported per month (per project, avg. rate)")
    ax.set_ylabel("Number of projects")
    ax.set_title("Distribution of average issue-reporting rate")
    fig.tight_layout()
    fig.savefig(output_dir / "issues_per_month_histogram.png", dpi=150)
    plt.close(fig)

    trend = cohort_stats.copy()
    trend["cohortMonth"] = pd.to_datetime(trend["cohortMonth"])
    trend = trend[trend["numProjects"] >= 5]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trend["cohortMonth"], trend["medianIssuesPerMonth"], label="median")
    ax.plot(trend["cohortMonth"], trend["meanIssuesPerMonth"], label="mean", alpha=0.6)
    ax.set_xlabel("Project creation cohort (month)")
    ax.set_ylabel("Avg. issues reported per month")
    ax.set_title("Issue-reporting rate by project creation cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "issues_per_month_by_cohort.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results.csv.gz",
        help="Path to the compressed results CSV (default: results.csv.gz next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "stats",
        help="Directory to write CSV (and optionally plot) output to",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also generate PNG plots of the distribution and cohort trend",
    )
    parser.add_argument(
        "--top-stars",
        type=int,
        default=10000,
        help="Restrict the analysis to the N projects with the most stargazers, "
        "selected among the projects passing --min-commits-per-week "
        "(default: 10000; pass 0 to disable and use all projects)",
    )
    parser.add_argument(
        "--min-commits-per-week",
        type=float,
        default=10.0,
        help="Restrict the analysis to projects averaging at least this many "
        "commits/week over their lifetime, i.e. consistently active "
        "(default: 10; pass 0 to disable)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_stars = args.top_stars if args.top_stars > 0 else None
    min_commits_per_week = (
        args.min_commits_per_week if args.min_commits_per_week > 0 else None
    )

    print(f"Loading {args.input} ...")
    df = load_data(
        args.input, top_stars=top_stars, min_commits_per_week=min_commits_per_week
    )
    if min_commits_per_week is not None:
        print(f"Restricted to projects with >= {min_commits_per_week} commits/week.")
    if top_stars is not None:
        print(f"Restricted to top {top_stars} projects by stargazers.")
    print(f"Loaded {len(df)} projects with valid dates.")

    overall = summary_overall(df)
    overall.to_csv(args.output_dir / "summary_overall.csv", index=False)

    percentiles = issues_per_month_percentiles(df)
    percentiles.to_csv(args.output_dir / "issues_per_month_percentiles.csv", index=False)

    cohort = issues_per_month_by_cohort(df)
    cohort.to_csv(args.output_dir / "issues_per_month_by_cohort.csv", index=False)

    by_language = stats_by_language(df)
    by_language.to_csv(args.output_dir / "stats_by_language.csv", index=False)

    corr = activity_correlations(df)
    corr.to_csv(args.output_dir / "activity_correlations.csv")

    if args.plot:
        make_plots(df, cohort, args.output_dir)

    non_fork = df[~df["isFork"]]
    print("\n=== Overview (non-fork projects) ===")
    print(f"Projects analysed:        {len(non_fork)}")
    print(f"Median issues/month:      {non_fork['issuesPerMonth'].median():.3f}")
    print(f"Mean issues/month:        {non_fork['issuesPerMonth'].mean():.3f}")
    print(f"Median issue close rate:  {non_fork['issueCloseRate'].median():.3f}")
    print(f"Median commits/week:      {non_fork['commitsPerWeek'].median():.3f}")
    print(f"\nOutput written to {args.output_dir}/")


if __name__ == "__main__":
    main()
