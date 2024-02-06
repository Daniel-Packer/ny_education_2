from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, colormaps, colors
import seaborn as sns
import datetime

root_path = Path().resolve().parents[1]
enrollment_2022_path = root_path / "data" / "enrollment_2022"
map_path = root_path / "data" / "NYS_Civil_Boundaries"
image_path = root_path / "images"


def load_data(year):
    """
    Loads combined dataframe of total enrollment, demographic factors, and BOCES/NRC data for a given year, dropping rows that don't correspond to a school.
    """
    enrollment_data = pd.read_csv(enrollment_2022_path / "enrollment_2022.csv")
    demographic_data = pd.read_csv(
        enrollment_2022_path / "demographic_factors_2022.csv"
    )
    boces_nrc_data = pd.read_csv(enrollment_2022_path / "boces_nrc_2022.csv")

    data = (
        demographic_data[demographic_data["YEAR"] == year]
        .merge(
            boces_nrc_data[boces_nrc_data["YEAR"] == year],
            how="left",
            left_on="ENTITY_CD",
            right_on="ENTITY_CD",
        )
        .merge(
            enrollment_data[["YEAR", "ENTITY_CD", "K12"]].query("YEAR == @year"),
            how="left",
            left_on="ENTITY_CD",
            right_on="ENTITY_CD",
        )
        .dropna(subset=["SCHOOL_NAME"])
    )
    return data


def get_exposure(data, county_name, pop_1, pop_2, renormalize=True):
    match county_name:
        case "ALL" | "NY STATE":
            county_data = data
        case "NYC":
            county_data = data[
                data["COUNTY_NAME"].isin(
                    ["BRONX", "KINGS", "NEW YORK", "QUEENS", "RICHMOND"]
                )
            ]
        case _:
            county_data = data[data["COUNTY_NAME"] == county_name]
    p_A = county_data[pop_1] / county_data[pop_1].sum()
    prop_B = county_data[pop_2] / county_data["K12"]
    score = (prop_B * p_A).sum()
    if renormalize:
        score = score * (county_data["K12"].sum() / county_data[pop_2].sum())
    return score


def get_exposure_mat(data, county_name, pop_list, renormalize=True):
    E = np.zeros((len(pop_list), len(pop_list)))
    for i, pop_1 in enumerate(pop_list):
        for j, pop_2 in enumerate(pop_list):
            E[i, j] = get_exposure(data, county_name, pop_1, pop_2, renormalize)
    return E


def plot_exposure_matrices(
    data, county_names, racial_pops, renormalize=True, filename=None
):
    racial_pops_prefixed = [f"NUM_{pop}" for pop in racial_pops]
    fig, axs = plt.subplots(3, 3, figsize=(16, 12), layout="compressed")
    axs = axs.flatten()

    vmin, vmax = (0.0, 1.0) if not renormalize else (0.3333, 3.0)
    center = None if not renormalize else None
    cmap = "plasma" if not renormalize else colormaps["seismic"]
    cbar_ax = fig.add_axes([0.90, 0.10, 0.025, 0.8])

    for ax, county_name in zip(axs, county_names[: len(axs)]):
        E = get_exposure_mat(
            data, county_name, racial_pops_prefixed, renormalize=renormalize
        )
        sns.heatmap(
            E,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            xticklabels=racial_pops,
            yticklabels=racial_pops,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            center=center,
            cbar_ax=cbar_ax,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax) if renormalize else None,
            cbar_kws={"ticks": [1.0]} if renormalize else None,
        )
        ax.set(title=county_name, aspect="equal")

    cbar_ax.set_ylabel("Average Class Exposure", {"fontsize": 14})
    if renormalize:
        cbar_ax.set_yticks([0.3333, 1.0, 3.0])
        cbar_ax.set_yticklabels(["<1/3", "1", ">3"])
        plt.suptitle("Renormalized Class Membership")
    else:
        plt.suptitle("Average Class Racial Makeups")

    filename = (
        filename
        if filename
        else (
            f"normalized_exposure_matrices"
            if renormalize
            else "unnormalized_exposure_matrices"
        )
    )
    filename = f"{filename}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(image_path / filename, bbox_inches="tight")
