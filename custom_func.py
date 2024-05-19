import pandas as pd


def preprocess_student_report(student_report_df, 
                              score_grade_file_path = "Data/score_grade.csv", 
                              subject_weight_file_path = "Data/subject_weight_values.csv"
                              ):
    
    # Read the files
    student_report_df = student_report_df
    score_grade_df = pd.read_csv(score_grade_file_path)
    subject_weight_values_df = pd.read_csv(subject_weight_file_path)

    
    # Get all columns from the student report that end with '_subject_score' or 'coursework_score'
    score_columns = [col for col in student_report_df.columns if col.endswith('_subject_score') or col.endswith('coursework_score')]

    # Create a dynamic weight mapping dictionary based on subject_weight_values_df
    weight_mapping = {col: col.replace('_subject_score', '_subject_weight_value').replace('coursework_score', 'coursework_weight_value(KK)') for col in score_columns}

    # Ensure the weight mapping only includes keys that exist in the subject_weight_values_df columns
    weight_mapping = {k: v for k, v in weight_mapping.items() if v in subject_weight_values_df.columns}

    # Function to map scores to grades and CGPA
    def get_grade_and_cgpa(score, score_grade_df):
        for index, row in score_grade_df.iterrows():
            min_score, max_score = map(int, row["marks"].split('-'))
            if min_score <= score <= max_score:
                return row["grade"], row["overall_cgpa"]
        return None, None

    # Function to calculate subject mark based on CGPA and weight value
    def calculate_subject_mark(cgpa, weight_value):
        return round(cgpa * (weight_value / 100), 2)

    # Add new columns for grades, CGPA, and subject marks
    for col in score_columns:
        grade_col = col.replace("_score", "_grade")
        cgpa_col = col.replace("_score", "_cgpa")
        mark_col = col.replace("_score", "_mark")
        
        student_report_df[grade_col] = student_report_df[col].apply(lambda x: get_grade_and_cgpa(x, score_grade_df)[0])
        student_report_df[cgpa_col] = student_report_df[col].apply(lambda x: round(get_grade_and_cgpa(x, score_grade_df)[1], 2))
        
        # Only calculate the subject mark if the corresponding weight value exists
        if col in weight_mapping:
            weight_value = subject_weight_values_df[weight_mapping[col]].values[0]
            student_report_df[mark_col] = student_report_df[cgpa_col].apply(lambda x: calculate_subject_mark(x, weight_value))

    # Calculate the overall CGPA without saving total_subject_mark
    mark_columns = [col for col in student_report_df.columns if col.endswith('_mark')]
    
    # student_report_df['cgpa'] = (student_report_df[mark_columns].sum(axis=1) / (student_report_df['student_current_semester'] + 1)).round(2)
    student_report_df['cgpa'] = (student_report_df[mark_columns].sum(axis=1) / 4).round(2)

    # # Save the updated DataFrame to a new CSV file
    # student_report_df.to_csv("updated_student_report_1.csv", index=False)
    
    return student_report_df
    
    
    