#Contains the implementation of the SpanBERT-based model, including the span scoring mechanism.
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class ExtractSpans(nn.Module):
    def __init__(self, sort_spans):
        super(ExtractSpans, self).__init__()
        self.sort_spans = sort_spans

    def forward(self, span_scores, candidate_starts, candidate_ends, num_output_spans, max_sentence_length):
        span_scores = span_scores.squeeze(0)
        candidate_starts = candidate_starts.squeeze(0)
        candidate_ends = candidate_ends.squeeze(0)
        num_output_spans = num_output_spans.item()

        num_sentences = span_scores.size(0)
        num_input_spans = span_scores.size(1)
        max_num_output_spans = num_output_spans

        output_span_indices = torch.zeros((num_sentences, max_num_output_spans), dtype=torch.int32)

        for l in range(num_sentences):
            sorted_indices = torch.argsort(span_scores[l], descending=True)
            top_span_indices = []
            end_to_earliest_start = {}
            start_to_latest_end = {}

            current_span_index = 0
            num_selected_spans = 0
            while num_selected_spans < num_output_spans and current_span_index < num_input_spans:
                i = sorted_indices[current_span_index].item()
                any_crossing = False
                start = candidate_starts[l, i].item()
                end = candidate_ends[l, i].item()
                for j in range(start, end + 1):
                    if (j in start_to_latest_end and j > start and start_to_latest_end[j] > end) or \
                       (j in end_to_earliest_start and j < end and end_to_earliest_start[j] < start):
                        any_crossing = True
                        break
                if not any_crossing:
                    if self.sort_spans:
                        top_span_indices.append(i)
                    else:
                        output_span_indices[l, num_selected_spans] = i
                    num_selected_spans += 1
                    if start not in start_to_latest_end or end > start_to_latest_end[start]:
                        start_to_latest_end[start] = end
                    if end not in end_to_earliest_start or start < end_to_earliest_start[end]:
                        end_to_earliest_start[end] = start
                current_span_index += 1

            if self.sort_spans:
                top_span_indices.sort(key=lambda i: (candidate_starts[l, i].item(), candidate_ends[l, i].item()))
                for i in range(num_output_spans):
                    output_span_indices[l, i] = top_span_indices[i]

            for i in range(num_selected_spans, max_num_output_spans):
                output_span_indices[l, i] = output_span_indices[l, 0]

        return output_span_indices

class SpanBERTCorefModel(nn.Module):
    def __init__(self, config):

        super(SpanBERTCorefModel, self).__init__()
        self.bert = self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        self.extract_spans = ExtractSpans(sort_spans=True)

        self.config = config

        # Feedforward networks for mention scoring and pair scoring
        self.ffnn_m = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 3, 150),  # 3: start, end, attention
            nn.ReLU(),
            nn.Linear(150, 1)
        )

        self.ffnn_c = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 6, 150),
            # features from two spans (gx, gy) and hand-crafted features
            nn.ReLU(),
            nn.Linear(150, 1)
        )

        self.max_span_length = config.MAX_TRAIN_LEN
        self.max_span_length = config.MAX_SPAN_WIDTH

    def __prepare_spans_candidates(self, sentence_map, num_words):

        device = sentence_map.device

        flattened_sentence_indices = sentence_map
        candidate_starts = torch.arange(num_words).unsqueeze(1).repeat(1, self.max_span_length)  # [num_words, max_span_width]
        candidate_ends = candidate_starts + torch.arange(self.max_span_length).unsqueeze(
            0)  # [num_words, max_span_width]

        candidate_starts = candidate_starts.to(device)
        candidate_ends = candidate_ends.to(device)

        candidate_start_sentence_indices = flattened_sentence_indices[candidate_starts]  # [num_words, max_span_width]
        # Ensure candidate ends do not exceed the maximum word index (num_words - 1)
        candidate_ends_clipped = torch.min(candidate_ends, torch.tensor(num_words - 1))
        candidate_end_sentence_indices = flattened_sentence_indices[
            candidate_ends_clipped]  # [num_words, max_span_width]

        candidate_mask = (candidate_ends < num_words) & (
                    candidate_start_sentence_indices == candidate_end_sentence_indices)  # [num_words, max_span_width]
        flattened_candidate_mask = candidate_mask.view(-1)  # [num_words * max_span_width]

        # Flatten the tensors
        flattened_candidate_starts = candidate_starts.view(-1)
        flattened_candidate_ends = candidate_ends.view(-1)
        flattened_candidate_sentence_indices = candidate_start_sentence_indices.view(-1)

        # Apply boolean mask to filter the flattened tensors
        candidate_starts = flattened_candidate_starts[flattened_candidate_mask]
        candidate_ends = flattened_candidate_ends[flattened_candidate_mask]
        candidate_sentence_indices = flattened_candidate_sentence_indices[flattened_candidate_mask]

        return candidate_starts, candidate_ends, candidate_sentence_indices

    def __get_span_emb(self, head_emb, context_outputs, span_starts, span_ends, num_words):
        span_emb_list = []

        span_start_emb = context_outputs[span_starts]
        span_emb_list.append(span_start_emb)

        span_end_emb = context_outputs[span_starts]  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts

        if self.config.USE_FEATURES:
            span_width_index = span_width - 1  # [k]
            span_embedding_module = SpanEmbeddingModule(self.config)
            # TODO: returnin NOne
            span_width_emb = span_embedding_module(span_width_index)
            span_emb_list.append(span_width_emb)

        if self.config.MODEL_HEADS:
            embedding_size = context_outputs.size(1)
            scorer = MentionWordScorer(self.config, embedding_size=embedding_size)
            mention_word_scores = scorer(context_outputs, span_starts, span_ends)
            span_emb_list.append(mention_word_scores)

        span_emb = torch.cat(span_emb_list, dim=1)  # [k, emb]

        return span_emb

    def __get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):

        same_start = (labeled_starts.unsqueeze(1) == candidate_starts.unsqueeze(0))
        same_end = (labeled_ends.unsqueeze(1) == candidate_ends.unsqueeze(0))

        same_span = same_start & same_end
        true_indices = torch.nonzero(same_span).squeeze()

        candidate_labels = torch.matmul(labels.unsqueeze(0).to(torch.float32), same_span.to(torch.float32))

        return candidate_labels.squeeze(0)

    def __flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = emb.size(0)
        max_sentence_length = emb.size(1)

        emb_rank = len(emb.shape)

        if emb_rank == 2:
            flattened_emb = emb.view(num_sentences * max_sentence_length)
        elif emb_rank == 3:
            flattened_emb = emb.view(num_sentences * max_sentence_length, emb.size(2))
        else:
            raise ValueError(f"Unsupported rank: {emb_rank}")

        # Flatten the mask and apply it
        flattened_mask = text_len_mask.view(num_sentences * max_sentence_length).bool()
        return flattened_emb[flattened_mask]

    def forward(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
        # Pass the document through the transformer encoder
        device = input_ids.device
        transformer_outputs = self.bert(input_ids=input_ids,
                                        attention_mask=input_mask)

        self.extract_spans.to(device)

        # Extract the hidden states from the last layer of the transformer
        sequence_output = transformer_outputs.last_hidden_state
        sequence_output = self.__flatten_emb_by_sentence(sequence_output, input_mask)
        num_words = sequence_output.size(0) # num_owrds sempre da 128, mas pode ter menos palavras e ai como faz?

        # You can now compute span representations (gx and gy) and feed them into FFNNs

        # first: prepare candidate spans for extraction, ensuring they are valid and
        #  fall within sentence boundaries
        candidate_starts, candidate_ends, candidate_sentence_indices = self.__prepare_spans_candidates(sentence_map,\
                                                                                                       num_words)

        candidate_cluster_ids = self.__get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates]
        candidate_span_emb = self.__get_span_emb(sequence_output, sequence_output, candidate_starts,
                                               candidate_ends, num_words)  # [num_candidates, emb]

        embedding_size = candidate_span_emb.size(1)  # span_emb shape is [k, embedding_dim], so embedding_dim = span_emb.size(1)
        model_mention_scorer = MentionScoreCalculator(self.config, embedding_size)
        model_mention_scorer = model_mention_scorer.to(device)

        candidate_mention_scores = model_mention_scorer(candidate_span_emb, candidate_starts, candidate_ends)
        candidate_mention_scores = torch.squeeze(candidate_mention_scores, 1)

        # beam size
        k = torch.min(torch.tensor(3900),
                      torch.floor(torch.tensor(num_words, dtype=torch.float32) * self.config.TOP_SPAN_RATIO).to(
                          torch.int32)).to(device)
        c = torch.min(torch.tensor(self.config.MAX_TOP_ANTECEDENTS), k).to(device)

        # pull from beam
        top_span_indices = self.extract_spans(candidate_mention_scores, candidate_starts, candidate_ends, k, num_words)
        #top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
        #                                           tf.expand_dims(candidate_starts, 0),
        #                                           tf.expand_dims(candidate_ends, 0),
        #                                           tf.expand_dims(k, 0),
        #                                           num_words,
        #                                           True)  # [1, k]

        return sequence_output




class SpanEmbeddingModule(nn.Module):
    def __init__(self, config):
        super(SpanEmbeddingModule, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        self.span_width_embeddings = nn.Parameter(
                torch.randn(config.MAX_SPAN_WIDTH, config.FEATURE_SIZE) * 0.02
            )

    def forward(self, span_width):
        # Calculate span width index (0-based)
        span_width_index = span_width - 1  # [k]

        device = span_width.device
        self.span_width_embeddings.data = self.span_width_embeddings.data.to(device)

        # Gather span width embeddings
        span_width_emb = self.span_width_embeddings[span_width_index]  # [k, emb]

        # Apply dropout to the span width embeddings
        span_width_emb = self.dropout(span_width_emb)

        return span_width_emb


class MentionWordScorer(nn.Module):
    def __init__(self, config, embedding_size):
        super(MentionWordScorer, self).__init__()
        self.config = config

        # Projection layer for word attention
        self.word_attn_proj = nn.Linear(embedding_size, 1)
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

    def forward(self, encoded_doc, span_starts, span_ends):
        num_words = encoded_doc.size(0)  # T: Number of words in the document
        num_c = span_starts.size(0)  # NC: Number of candidate spans


        # Create a document range tensor [T] and tile it for each candidate span [num_c, T]
        device = encoded_doc.device
        doc_range = torch.arange(0, num_words).unsqueeze(0).repeat(num_c, 1).to(device)  # [num_c, T]
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        self.word_attn_proj = self.word_attn_proj.to(device) 

        # Create mention mask: True for words within the span, False otherwise
        mention_mask = (doc_range >= span_starts.unsqueeze(1)) & (doc_range <= span_ends.unsqueeze(1))  # [num_c, T]
        mention_mask = mention_mask.float().to(device) 

        # Word attention using a linear projection on encoded_doc [T, emb]
        word_attn = self.word_attn_proj(encoded_doc).squeeze(1).to(device)  # [T]

        # Apply mask and softmax for attention scores
        # Convert mention_mask to float, take log for stability
        mention_mask = mention_mask.float()  # [num_c, T]
        mention_word_attn = F.softmax(torch.log(mention_mask + 1e-10).to(device) + word_attn.unsqueeze(0), dim=-1)  # [num_c, T]

        return mention_word_attn

class MentionScoreCalculator(nn.Module):
    def __init__(self, config, embedding_dim):
        super(MentionScoreCalculator, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        # Define FFNN layers
        self.ffnn = nn.Sequential(
            nn.Linear(embedding_dim, config.FFNN_SIZE),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(config.FFNN_SIZE, config.FFNN_SIZE),
                nn.ReLU()
            ) for _ in range(config.FFNN_SIZE - 1)],
            nn.Linear(config.FFNN_SIZE, 1)
        )

        if self.config.USE_PRIOR:
            self.span_width_emb = nn.Parameter(
                torch.randn(self.config.MAX_SPAN_WIDTH, self.config.FEATURE_SIZE) * 0.02)
            self.width_ffnn = nn.Sequential(
                nn.Linear(config.FEATURE_SIZE, config.FFNN_SIZE),
                nn.ReLU(),
                *[nn.Sequential(
                    nn.Linear(config.FFNN_SIZE, config.FFNN_SIZE),
                    nn.ReLU()
                ) for _ in range(config.FFNN_DEPTH - 1)],
                nn.Linear(config.FFNN_SIZE, 1)
            )

    def forward(self, span_emb, span_starts, span_ends):
        # Compute span scores
       
        span_scores = self.ffnn(self.dropout(span_emb))

        if self.config.USE_PRIOR:
            # Compute span width index
            span_width_index = span_ends - span_starts

            # Compute width scores
            width_scores = self.width_ffnn(self.span_width_emb)
            width_scores = width_scores[span_width_index]

            # Add width scores to span scores
            span_scores += width_scores

        return span_scores
