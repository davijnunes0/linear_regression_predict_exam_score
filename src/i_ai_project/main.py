import pandas as pd

from utils.paths import path_resolve
from data.loading import load_encodings, read_csv
from data.encoding import convert_to_numeric
from analysis.statistics import pearson_correlation
from analysis.construction_of_correlation_matrix import build_correlation_matrix, plot_target_correlations,plot_correlation_matrix,  get_target_correlations, plot_feature_vs_target

def main():

    DATA_PATH = path_resolve(2, "Student Performance Factors.csv")
    data_frame = pd.read_csv(DATA_PATH)
    data_frame.info()

    encodings = load_encodings(path_resolve(2, "encodings.json"))

    rows = read_csv(DATA_PATH)

    numeric_columns = convert_to_numeric(
        rows,
        encodings
    )

    x = numeric_columns["Hours_Studied"]


    y = numeric_columns["Exam_Score"]

    print()

    correlation = pearson_correlation(x,y)


    correlation_matrix, columns = build_correlation_matrix(numeric_columns)

    target_correlations = get_target_correlations(
        correlation_matrix,
        columns,
        "Exam_Score"
    )

    plot_target_correlations(
        target_correlations,
        "Exam_Score"
    )


    plot_correlation_matrix(
        correlation_matrix,
        columns,
        "Exam_Score"
    )

    fig = plot_feature_vs_target(
        numeric_columns["Attendance"],
        numeric_columns["Exam_Score"],
        "Attendance",
        "Exam_Score"
    )


if __name__ == "__main__":
    main()
