import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp


def main():
    data_source_path = "C:/Users/Rahman/Downloads/dclp2_data_935_943_961.csv"
    all_junctions = pd.read_csv(data_source_path)
    all_junctions = data_by_cid(all_junctions, 935)

    all_junctions["cycle_start_time"] = pd.to_datetime(
        all_junctions["cycle_start_time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise"
    )

    hourly_data = hourly_resampling(all_junctions)    
    split_datetime = pd.Timestamp("2024-11-11 23:59:59")
    metrics = ["vehicle_count", "ds"]

    case_A, case_B = split_AB_cases(hourly_data, split_datetime)
    case_A = case_A[case_A["plan_name"] == "PLNAP"]
    case_B = case_B[case_B["plan_name"] == "PLN02"]

    metric_to_statistics = {}

    for metric in metrics:
        case_A_normality = test_normality(case_A[metric])
        case_B_normality = test_normality(case_B[metric])

        case_A_is_normal = case_A_normality["p_value"] > 0.05
        case_B_is_normal = case_B_normality["p_value"] > 0.05

        if case_A_is_normal and case_B_is_normal:
            parametric_stats = parametric_statistics(case_A[metric], case_B[metric])
            metric_to_statistics[metric] = parametric_stats
        else:
            non_parametric_stats = non_parametric_statistics(case_A[metric], case_B[metric])
            metric_to_statistics[metric] = non_parametric_stats

    for metric, statistic in metric_to_statistics.items():
        print(f"Metric: {metric}")

        if "t_statistic" in statistic:
            t_statistic = statistic["t_statistic"]
            p_value = statistic["p_value"]
            print(f"t-statistic: {t_statistic}")
            print(f"p-value: {p_value}")
        else:
            u_statistic = statistic["u_statistic"]
            u_p_value = statistic["u_p_value"]
            ks_statistic = statistic["ks_statistic"]
            ks_p_value = statistic["ks_p_value"]
            print(f"u-statistic: {u_statistic}")
            print(f"u-p-value: {u_p_value}")
            print(f"ks-statistic: {ks_statistic}")
            print(f"ks-p-value: {ks_p_value}")

def data_by_cid(df, cid):
    output = df[df["cid"] == cid]

    return output

def hourly_resampling(df):
    df_set_index = df.set_index("cycle_start_time")

    df_set_index = df_set_index.sort_index()

    df_resampled = df_set_index.resample("h").agg({
        "plan_name": "first",  
        "vehicle_count": "sum",    
        "ds": "mean"
    }).reset_index()

    df_resampled = df_resampled.sort_values(by="cycle_start_time", ascending=True)

    df_dropped_NA = df_resampled.dropna()

    return df_dropped_NA

def split_AB_cases(df, split_datetime):
    case_A = df[df["cycle_start_time"] <= split_datetime]
    case_B = df[df["cycle_start_time"] > split_datetime]

    return case_A, case_B

def test_normality(df):
    res = {}

    shapiro_statistic, p_value = shapiro(df)

    res["shapiro_statistic"] = shapiro_statistic
    res["p_value"] = p_value

    return res

def parametric_statistics(case_A, case_B):
    results = {}

    # calculate variance assuming case_A and case_B is pandas dataframe colum
    variance_A = case_A.var()
    variance_B = case_B.var()

    if variance_A != variance_B:
        t_statistic, p_value = ttest_ind(case_A, case_B, alternative='greater', equal_var=False)
    else:
        t_statistic, p_value = ttest_ind(case_A, case_B, alternative='greater', equal_var=True)

    results["t_statistic"] = t_statistic
    results["p_value"] = p_value

    return results

def non_parametric_statistics(case_A, case_B):
    res = {}

    u_statistic, u_p_value = mannwhitneyu(case_A, case_B, alternative='greater')
    ks_statistic, ks_p_value = ks_2samp(case_A, case_B, alternative='greater')
    
    res["u_statistic"] = u_statistic
    res["u_p_value"] = u_p_value
    res["ks_statistic"] = ks_statistic
    res["ks_p_value"] = ks_p_value
    
    return res


if __name__ == "__main__":
    main()