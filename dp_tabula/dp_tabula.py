import os
import warnings
import json
import typing as tp
import logging

import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments,
                          AutoConfig)

from dp_tabula.tabula_dataset import tabulaDataset, tabulaDataCollator
from dp_tabula.tabula_start import tabulaStart, CategoricalStart, ContinuousStart, RandomStart
from dp_tabula.dp_tabula_trainer import dp_tabulaTrainer
from dp_tabula.tabula_utils import _array_to_dataframe, _get_column_distribution, _convert_tokens_to_text, \
    _convert_text_to_tabular_data


class DP_Tabula:
    """
    The tabula class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(self, llm: str, experiment_dir: str = "trainer_tabula", epochs: int = 100,
                 batch_size: int = 8, categorical_columns: list = [], 
                 use_dp: bool = False, clip_coeff: float = 1.0, sigma: float = 1.0, 
                 micro_batch_size: int = 1, learning_rate: int = 1e-3, pretrain_epochs: int = 10, **train_kwargs):
        """ Initializes tabula.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used as the basis of our model
            experiment_dir:  Directory where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            categorical_columns: List of categorical columns in the data
            use_dp: Boolean indicating whether to enable DP-SGD
            clip_coeff: Gradient clipping coefficient for DP-SGD
            sigma: Noise multiplier for DP-SGD
            micro_batch_size: Size of micro-batches for DP-SGD
            train_kwargs: Additional hyperparameters added to the TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = AutoConfig.from_pretrained(self.llm)
        self.model = AutoModelForCausalLM.from_config(self.config)

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.categorical_columns = categorical_columns
        self.train_hyperparameters = train_kwargs
        
        # DP-related parameters
        self.use_dp = use_dp
        self.clip_coeff = clip_coeff
        self.sigma = sigma
        self.micro_batch_size = micro_batch_size
        self.pretrain_epochs = pretrain_epochs
        self.learning_rate = learning_rate

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def encode_categorical_column(self, data: pd.DataFrame):
        self.label_encoder_list = []
        for column_index, column in enumerate(data.columns):            
            if column in self.categorical_columns:        
                label_encoder = preprocessing.LabelEncoder()
                data[column] = data[column].astype(str)
                label_encoder.fit(data[column])
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(data[column])
                data[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
        return data
    

    def decode_categorical_column(self, data: pd.DataFrame):

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            allowed_values = list(range(len(le.classes_)))
            
            # delete rows that should generate numeric value but generate other data type
            data[self.label_encoder_list[i]['column']] = pd.to_numeric(data[self.label_encoder_list[i]['column']], errors='coerce')
            data = data.dropna(subset=[self.label_encoder_list[i]['column']])

            # delete rows that generate category that is out of boundary
            data[self.label_encoder_list[i]['column']] = data[self.label_encoder_list[i]['column']].astype(float)
            data = data[data[self.label_encoder_list[i]['column']].isin(allowed_values)]


        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            data[self.label_encoder_list[i]["column"]] = data[self.label_encoder_list[i]["column"]].astype(int)
            data[self.label_encoder_list[i]["column"]] = le.inverse_transform(data[self.label_encoder_list[i]["column"]])
            
        return data

    def fit(self, data: tp.Union[pd.DataFrame, np.ndarray], column_names: tp.Optional[tp.List[str]] = None,
            conditional_col: tp.Optional[str] = None, resume_from_checkpoint: tp.Union[bool, str] = False) -> dp_tabulaTrainer:
        """
        Fine-tune tabula using tabular data with an initial pre-training phase without DP, followed by a DP phase.
        """
        # Convert data to DataFrame and prepare dataset
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)
        if self.categorical_columns:
            df = self.encode_categorical_column(df)
        tabula_ds = tabulaDataset.from_pandas(df)
        tabula_ds.set_tokenizer(self.tokenizer)

        dp_train_args = TrainingArguments(
            output_dir=self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            save_strategy="no",
            report_to="none",
            learning_rate=self.learning_rate,
            **self.train_hyperparameters
        )

        dp_trainer = dp_tabulaTrainer(
            model=self.model,
            args=dp_train_args,
            train_dataset=tabula_ds,
            tokenizer=self.tokenizer,
            data_collator=tabulaDataCollator(self.tokenizer),
            use_dp=True,
            clip_coeff=self.clip_coeff,
            sigma=self.sigma,
            micro_batch_size=self.micro_batch_size
        )

        # Initialize parameters for privacy tracking
        delta = 1e-5
        q = self.batch_size / len(tabula_ds)
        self.total_updates = 0

        logging.info("Starting DP-SGD training phase...")
        dp_trainer.train(resume_from_checkpoint=False)

        return dp_trainer

    def sample(self, n_samples: int,
               start_col: tp.Optional[str] = "", start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
               temperature: float = 0.7, k: int = 100, max_length: int = 100, device: str = "cuda") -> pd.DataFrame:
        """ Generate synthetic tabular data samples

        Args:
            n_samples: Number of synthetic samples to generate
            start_col: Feature to use as starting point for the generation process. If not given, the target
             learned during the fitting is used as starting point
            start_col_dist: Feature distribution of the starting feature. Should have the format
             "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns.
             If not given, the target distribution learned during the fitting is used as starting point
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process
            k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """
        tabula_start = self._get_start_sampler(start_col, start_col_dist)

        # Move model to device
        self.model.to(device)

        # Init empty DataFrame for the generated samples
        df_gen = pd.DataFrame(columns=self.columns)

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            while n_samples > df_gen.shape[0]:
                start_tokens = tabula_start.get_start_tokens(k)
                start_tokens = torch.tensor(start_tokens).to(device)

                # Generate tokens
                tokens = self.model.generate(input_ids=start_tokens, max_length=max_length,
                                             do_sample=True, temperature=temperature, pad_token_id=50256)

                # Convert tokens back to tabular data
                text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                df_gen = _convert_text_to_tabular_data(text_data, df_gen)

                # Remove rows with flawed numerical values
                for i_num_cols in self.num_cols:
                    df_gen = df_gen[pd.to_numeric(df_gen[i_num_cols], errors='coerce').notnull()]

                df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                # Remove rows with missing values
                df_gen = df_gen.drop(df_gen[df_gen.isna().any(axis=1)].index)

                # Update process bar
                pbar.update(df_gen.shape[0] - already_generated)
                already_generated = df_gen.shape[0]


        df_gen = df_gen.reset_index(drop=True)

        if self.categorical_columns == []:
            return df_gen.head(n_samples)
        else:
            df_inversed = self.decode_categorical_column(df_gen.head(n_samples))
            return df_inversed

    def tabula_sample(self, starting_prompts: tp.Union[str, list[str]], temperature: float = 0.7, max_length: int = 100,
                     device: str = "cuda") -> pd.DataFrame:
        """ Generate synthetic tabular data samples conditioned on a given input.

        Args:
            starting_prompts: String or List of Strings on which the output is conditioned.
             For example, "Sex is female, Age is 26"
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process.
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.

        self.model.to(device)
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        generated_data = []

        # Generate a sample for each starting point
        for prompt in tqdm(starting_prompts):
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            # Generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=max_length,
                                      do_sample=True, temperature=temperature, pad_token_id=50256)
            generated_data.append(torch.squeeze(gen))

        # Convert Text back to tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(decoded_data, pd.DataFrame(columns=self.columns))

        return df_gen

    def save(self, path: str):
        """ Save tabula Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """ Load fine-tuned model

        Load the weights of a fine-tuned large language model into the tabula pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """ Load tabula class

        Load trained tabula model from directory.

        Args:
            path: Directory where tabula model is saved

        Returns:
            New instance of tabula loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_tabula model instance
        tabula = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(tabula, k, v)

        # Load model weights
        tabula.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return tabula

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None):
        assert conditional_col is None or isinstance(conditional_col, str), \
            f"The column name has to be a string and not {type(conditional_col)}"
        assert conditional_col is None or conditional_col in df.columns, \
            f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(self, start_col: tp.Optional[str],
                           start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]]) -> tabulaStart:
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} was given, but no corresponding distribution.")
        if start_col_dist is not None and not start_col:
            raise ValueError(f"Start column distribution {start_col} was given, the column name is missing.")

        assert start_col is None or isinstance(start_col, str), \
            f"The column name has to be a string and not {type(start_col)}"
        assert start_col_dist is None or isinstance(start_col_dist, dict) or isinstance(start_col_dist, list), \
            f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)
