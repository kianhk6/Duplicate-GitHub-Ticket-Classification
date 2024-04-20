import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
import configurations as cfg

DATASET_NAME = cfg.DATASET_NAME

class EmbeddingsGenerator:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
    
    def generate_embeddings(self, title, content=None):
        combined_text = f"{title} {content}" if pd.notna(content) else title
        embedding = self.model.encode(combined_text, convert_to_tensor=True)
        return embedding.cpu().numpy()

class DatasetManager:
    def __init__(self, dataset_name=DATASET_NAME):
        self.dataset_name = dataset_name
        self.df = pd.read_pickle(self.dataset_name)

    @staticmethod
    def typecast_df(df):
        df['Duplicated_issues'] = df['Duplicated_issues'].apply(lambda x: [int(i) for i in x])
        df["Issue_id"] = df["Issue_id"].astype('Int64')
        return df

class EmbeddingsPipeline:
    def __init__(self, dataset_name, model_name):
        self.dataset_manager = DatasetManager(dataset_name)
        self.embeddings_generator = EmbeddingsGenerator(model_name)
    
    def process(self):
        embeddings_filename = f'{DATASET_NAME}_embeddings_{self.embeddings_generator.model_name}'
        embeddings_filepath = f'../data/{embeddings_filename}'
        
        if not os.path.exists(embeddings_filepath):
            self.process_and_save_embeddings(embeddings_filename)
            print(f"Embeddings file created: {embeddings_filename}")
        else:
            print(f"Embeddings file already exists: {embeddings_filename}")
    
    def process_and_save_embeddings(self, filename):
        self.dataset_manager.df = DatasetManager.typecast_df(self.dataset_manager.df)
        self.dataset_manager.df['Duplicated_issues'] = self.dataset_manager.df['Duplicated_issues'].apply(tuple)
        
        # Sort dataframe by 'Duplicates_count' in descending order and select the top 100
        sorted_df = self.dataset_manager.df.sort_values(by='Duplicates_count', ascending=False).head(10000)
        
        # # Grab the duplicated issues of each row in the dataframe
        # duplicated_issues = sorted_df['Duplicated_issues'].apply(lambda x: [i for i in x if i not in sorted_df['Issue_id'].tolist()])

        # # Insert the duplicated issues into the dataframe
        # sorted_df['Duplicated_issues'] = duplicated_issues
        
        # Generate embeddings for the sorted and top selected dataset
        embeddings_df = pd.DataFrame()
        embeddings_df['Embedding'] = sorted_df.apply(
            lambda row: self.embeddings_generator.generate_embeddings(
                row['Title'], 
                '' if pd.isna(row.get('Content')) else row.get('Content')
            ), axis=1)
        embeddings_df['Issue_id'] = sorted_df['Issue_id']
        embeddings_df['Duplicated_issues'] = sorted_df['Duplicated_issues']
        
        # Reset index and save
        embeddings_df = embeddings_df.reset_index(drop=True)
        self.save_dataframe(embeddings_df, filename)

    def save_dataframe(self, df, filename):
        directory = os.path.dirname(f'../data/{filename}')
        os.makedirs(directory, exist_ok=True)
        df.to_pickle(f'../data/{filename}.pkl')

if __name__ == "__main__":
    embeddings_pipeline = EmbeddingsPipeline()
    embeddings_pipeline.process()
