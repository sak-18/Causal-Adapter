import torch
from torch import nn
import torch.nn.functional as F


PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    """
    Returns the token ID for a given string using the CLIP tokenizer.

    Args:
        tokenizer: CLIP tokenizer.
        string: String to be tokenized.

    Returns:
        Token ID for the given string.
    """
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    """
    Returns the token ID for a given string using the BERT tokenizer.

    Args:
        tokenizer: BERT tokenizer.
        string: String to be tokenized.

    Returns:
        Token ID for the given string.
    """
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    """
    Returns the embedding for a given CLIP token.

    Args:
        embedder: CLIP embedder.
        token: Token ID for which the embedding is to be fetched.

    Returns:
        Embedding for the given token.
    """
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    """
    Manages custom embeddings for placeholder tokens.

    Args:
        embedder: Model used for embedding.
        placeholder_strings: List of placeholder strings to be replaced with embeddings.
        presudo_words: List of presudo words for embeddings.
        initializer_words: List of words for initializing embeddings.
        per_image_tokens: Boolean to determine if per image tokens are used.
        num_vectors_per_token: Number of vectors per token.
        progressive_words: Boolean to determine if progressive words are used.
        **kwargs: Additional arguments.
    """
    def __init__(
            self,
            presudo_token_ids
    ):
        super().__init__()

        self.presudo_token_ids = presudo_token_ids

        self.embed_proj = torch.nn.Linear(768,768)
            
        
        #print(f'DEBUG EmbeddingManager init string_to_param_dict.keys: {self.string_to_param_dict.keys()}')

    def forward(
            self,
            input_ids,inputs_embeds,
    ):
        """ (replace the bert embeds with the stored embeddings e.t. ckpt)
        Replaces placeholder tokens in tokenized_text with corresponding embeddings.

        Args:
            tokenized_text: Tokenized input text.
            embedded_text: Embedded input text.

        Returns:
            Embedded text with placeholder tokens replaced by corresponding embeddings.
        """
        b, n, device = *input_ids.shape, input_ids.device
        for token_id in self.presudo_token_ids:
            # if token_id.clone().cpu() not in input_ids.clone().cpu():
            #     pass 
            placeholder_idx = torch.where(input_ids == token_id)
            placeholder_embedding = self.embed_proj(inputs_embeds[placeholder_idx])
            inputs_embeds[placeholder_idx] = placeholder_embedding

        
        return inputs_embeds 


    def load(self, ckpt_path):
        """
        Loads the state of the EmbeddingManager from a checkpoint file.

        Args:
            ckpt_path: Path to the checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        """
        Returns the squared norms of the embeddings.

        Returns:
            Tensor of squared norms of embeddings.
        """
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        """
        Returns the embedding parameters.

        Returns:
            Iterable of embedding parameters.
        """
        
        return self.string_to_param_dict.parameters()
        
    def embedding_to_coarse_loss(self):
        """
        Computes the loss between the optimized embeddings and the initial embeddings.

        Returns:
            Loss value.
        """
        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss



class Embed_control_manager(nn.Module):
    """
    Manages custom embeddings for placeholder tokens.

    Args:
        embedder: Model used for embedding.
        placeholder_strings: List of placeholder strings to be replaced with embeddings.
        presudo_words: List of presudo words for embeddings.
        initializer_words: List of words for initializing embeddings.
        per_image_tokens: Boolean to determine if per image tokens are used.
        num_vectors_per_token: Number of vectors per token.
        progressive_words: Boolean to determine if progressive words are used.
        **kwargs: Additional arguments.
    """
    def __init__(
            self,
            control_concept_ids
    ):
        super().__init__()

        self.control_concept_ids = control_concept_ids
        self.d = len(self.control_concept_ids)
       
        #print(f'DEBUG EmbeddingManager init string_to_param_dict.keys: {self.string_to_param_dict.keys()}')

    def forward(
            self,
            input_ids,inputs_embeds,attribute_cond=None
    ):
        """ (replace the bert embeds with the stored embeddings e.t. ckpt)
        Replaces placeholder tokens in tokenized_text with corresponding embeddings.

        Args:
            tokenized_text: Tokenized input text.
            embedded_text: Embedded input text.

        Returns:
            Embedded text with placeholder tokens replaced by corresponding embeddings.
        """
        b, n, device = *input_ids.shape, input_ids.device
        for i,token_id in enumerate(self.control_concept_ids):
            # if token_id.clone().cpu() not in input_ids.clone().cpu():
            #     pass 
            placeholder_idx = torch.where(input_ids == token_id)
            inputs_embeds[placeholder_idx] = inputs_embeds[placeholder_idx]+attribute_cond[:,i,:]
        
        return inputs_embeds 


    

    
    