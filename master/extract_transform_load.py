import pandas as pd
import configurations as cfg

test_filename = cfg.TEST_DATASET_FILE
train_filename = cfg.TRAIN_DATASET_FILE

class ETLProcessor:
    def __init__(self, dataset_filename, dataset_filename_pickle):
        self.dataset_filename = dataset_filename
        self.dataset_filename_pickle = dataset_filename_pickle
        self.duplicates_mapping = self.load_and_merge_test_train_files()

    def load_and_merge_test_train_files(self):
        test_df = pd.read_csv(test_filename)
        train_df = pd.read_csv(train_filename)
        combined_df = pd.concat([test_df, train_df])
        duplicates_mapping = {}
        for _, row in combined_df.iterrows():
            issue_id = row['Issue_id']
            duplicates = row['Duplicate']
            if pd.notnull(duplicates):
                duplicates_list = [int(d.strip()) for d in duplicates.split(',') if d.strip().isdigit()]
                for d in duplicates_list:
                    if issue_id in duplicates_mapping:
                        duplicates_mapping[issue_id].add(d)
                    else:
                        duplicates_mapping[issue_id] = {d}
                    if d in duplicates_mapping:
                        duplicates_mapping[d].add(issue_id)
                    else:
                        duplicates_mapping[d] = {issue_id}
        return duplicates_mapping

    def load_and_process_data(self):
        df = pd.read_csv(self.dataset_filename)
        df = self.typecast_df(df)
        df = self.identify_and_label_duplicates(df)
        df = self.clean_and_prepare_final_dataframe(df)
        self.save_dataframe(df)

    def typecast_df(self, df):
        df['Content'] = df['Title'] + ' ' + df['Description']
        df['Duplicated_issue'] = pd.to_numeric(df['Duplicated_issue'], errors='coerce').fillna(0).astype('Int64')
        df["Issue_id"] = df["Issue_id"].astype('Int64')
        return df

    def identify_and_label_duplicates(self, df):
        df['Issue_id'] = df['Issue_id'].astype(int)  # Ensure Issue_id is int for matching
        df['Duplicated_issues'] = df['Issue_id'].apply(lambda x: list(self.duplicates_mapping.get(x, [])))
        return df

    def clean_and_prepare_final_dataframe(self, df):
        df = df.drop(columns=['Duplicated_issue'])
        df.loc[(df['Duplicated_issues'].str.len() == 0) & (df['Resolution'] == 'DUPLICATE'), 'Resolution'] = 'NDUPLICATE'
        df['Duplicates_count'] = df['Duplicated_issues'].apply(len)
        return df

    def save_dataframe(self, df):
        df.to_pickle(self.dataset_filename_pickle)

if __name__ == "__main__":
    dataset_filename = cfg.DATASET_FILE
    dataset_filename_pickle = cfg.DATASET_PICKLE_FILE
    etl_processor = ETLProcessor(dataset_filename, dataset_filename_pickle)
    etl_processor.load_and_process_data()
