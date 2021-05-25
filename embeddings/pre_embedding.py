#Bert
import torch
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class Bert_Embedding():
    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_preBert = BertModel.from_pretrained('bert-base-uncased')
        self.model_preBert.eval()
        self.config = config

    def Pre_Bert_Embedding(self, textWords, reactionWords, model):
        token_claims = []
        token_claim_comms = []
        for text, comms in zip(textWords,reactionWords):
            #### text tokens
            marked_claim = "[CLS] " + text + " [SEP]"
            tokenized_text_claim = self.tokenizer.tokenize(marked_claim)
            indexed_tokens_claim = self.tokenizer.convert_tokens_to_ids(tokenized_text_claim)
            segments_ids_claim = [1]*len(indexed_tokens_claim)
            tokens_tensor_claim = torch.tensor([indexed_tokens_claim])
            segments_tensors_claim = torch.tensor([segments_ids_claim])
            token_claims.append((tokens_tensor_claim, segments_tensors_claim))

            #### claims+comments tokens
            token_comms = []
            for comm in comms:

                #marked_claim_comm = "[CLS] " + text + " [SEP]"
                marked_claim_comm = comm + " [SEP]"
                tokenized_text_comm = self.tokenizer.tokenize(marked_claim_comm)
                indexed_tokens_comm = self.tokenizer.convert_tokens_to_ids(tokenized_text_comm)
                segments_ids_comm = [0]*len(indexed_tokens_claim)+[1]*len(indexed_tokens_comm)
                tokens_tensor_comm = torch.tensor([indexed_tokens_claim+indexed_tokens_comm])
                segments_tensors_comm = torch.tensor([segments_ids_comm])
                token_comms.append((tokens_tensor_comm, segments_tensors_comm))
            token_claim_comms.append(token_comms)
        ### bert embeddings
        claim_embeds = []
        claim_comm_embeds = []
        claim_comm_words = []
        for token_claim, token_claim_comm in tqdm(zip(token_claims, token_claim_comms)):
            with torch.no_grad():
                hidden_states_claim, pooled_output_claim = model(token_claim[0], token_claim[1])
                claim_embeds.append(pooled_output_claim)
                claim_comm_embed = []
                claim_comm_words.append([])
                claim_comm_words[-1].append(hidden_states_claim[-1][0])
                for token_claim_comm_each in tqdm(token_claim_comm):
                    hidden_states_claim_comm, pooled_output_claim_comm = model(token_claim_comm_each[0], token_claim_comm_each[1])
                    len_claim = hidden_states_claim[-1][0].size()[0]
                    claim_comm_words[-1].append(hidden_states_claim_comm[-1][0][len_claim:,:])

                    claim_comm_embed.append(pooled_output_claim_comm)
                claim_comm_embeds.append(claim_comm_embed)

        return claim_embeds, claim_comm_embeds, claim_comm_words
    
    def Embed_truncate(self, claim_embeds, claim_comm_embeds, nums_comm):
            ###truncate & offset
            claim_embeds
            claim_comm_embeds_trunc = []

            for claim_comm_embed in claim_comm_embeds:
                claim_comm_embed_trunc = []
                if len(claim_comm_embed) < nums_comm:
                    for embed in claim_comm_embed:
                        claim_comm_embed_trunc.append(embed)

                    for i in range((nums_comm-len(claim_comm_embed))):
                        claim_comm_embed_trunc.append(torch.zeros(1, self.config.nums_bert))

                else:
                    sen_num = 0
                    for i in range(nums_comm):
                        claim_comm_embed_trunc.append(claim_comm_embed[i])

                claim_comm_embeds_trunc.append(torch.stack(claim_comm_embed_trunc, dim=1).squeeze())
            return claim_comm_embeds_trunc, claim_embeds