import torch
from torch import nn, optim


class CRF(nn.Module):
    def __init__(self, num_tags: int = 1):
        """
        conditional random field - pytorch module
        inspired by https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
        :param num_tags: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = num_tags + 2
        # set START_TAG and STOP_TAG to the last two tags
        self.START_TAG = self.tagset_size - 2
        self.STOP_TAG = self.tagset_size - 1
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000

    @staticmethod
    def _argmax(vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    @staticmethod
    # Compute log sum exp in a numerically stable way for the forward algorithm
    def _log_sum_exp(vec):
        max_score = vec[0, CRF._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(CRF._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        alpha = CRF._log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = CRF._argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        best_tag_id = CRF._argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def loss(self, sentence, tags):
        """
        :param sentence: a torch tensor of size (n, d) where n is sentence length, d is dimension
        :param tags: a tensor of size (n,) where n is sentence length = number of tags. tags are in the set {0, 1, ..., num_tags-1}
        :return: neg_log_likelihood
        """
        # neg_log_likelihood
        feats = sentence
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        """
        :param sentence: a torch tensor of size (n, d) where n is sentence length, d is dimension
        :return: score and best tag sequence
        """
        # Get the emission scores from the BiLSTM
        lstm_feats = sentence
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == "__main__":
    torch.manual_seed(1)
    # DATA
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    idx2word = list({word for sentence, tags in training_data for word in sentence})
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    print(word2idx)
    idx2tag = list({tag for sentence, tags in training_data for tag in tags})
    tag2idx = {tag: idx for idx, tag in enumerate(idx2tag)}
    print(tag2idx)


    def prepare_sentence(sentence):
        """
        :param sentence: list of words
        :return:
        """
        return torch.tensor([word2idx[word] for word in sentence])


    def prepare_tags(tags):
        """
        :param tags: list of tags
        :return:
        """
        return torch.tensor([tag2idx[tag] for tag in tags])

    # MODEL
    class Model(nn.Module):
        def __init__(
                self,
                num_words: int,
                num_tags: int,
                embedding_dim: int = 5,
                hidden_dim: int = 4,
        ):
            super(Model, self).__init__()
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(
                num_embeddings=num_words,
                embedding_dim=embedding_dim,
            )
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.linear = nn.Linear(
                in_features=2 * hidden_dim,  # bidirectional lstm
                out_features=num_tags + 2,  # add start and stop tag
            )
            self.crf = CRF(
                num_tags=num_tags,
            )

        def random_hidden(self):
            return (torch.randn(2, 1, self.hidden_dim),
                    torch.randn(2, 1, self.hidden_dim))

        def loss(self, sentence, tags):
            """
            :param sentence: a tensor of size (n,) where n is the sentence length. each entry is a number representing word. (0, 1, 2, ...)
            :param tags: a tensor of size (n,) where n is sentence length = number of tags. tags are in the set {0, 1, ..., num_tags-1}
            :return:
            """
            # embedding
            embedding_out = self.embedding(sentence)
            # lstm
            lstm_in = embedding_out.view(1, len(sentence), self.embedding_dim)
            hidden = self.random_hidden()
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            # linear
            linear_in = lstm_out.view(len(sentence), 2 * self.hidden_dim)
            linear_out = self.linear(linear_in)
            # crf
            crf_in = linear_out
            crf_loss_out = self.crf.loss(crf_in, tags)
            return crf_loss_out

        def forward(self, sentence):
            """
            :param sentence: a tensor of size (n,) where n is the sentence length. each entry is a number representing word. (0, 1, 2, ...)
            :return:
            """
            # embedding
            embedding_out = self.embedding(sentence)
            # lstm
            lstm_in = embedding_out.view(1, len(sentence), self.embedding_dim)
            hidden = self.random_hidden()
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            # linear
            linear_in = lstm_out.view(len(sentence), 2 * self.hidden_dim)
            linear_out = self.linear(linear_in)
            # crf
            crf_in = linear_out
            score, tags = self.crf.forward(crf_in)
            return tags


    model = Model(
        num_words=len(idx2word),
        num_tags=len(idx2tag),
        embedding_dim=5,
        hidden_dim=2,
    )

    with torch.no_grad():
        sentence, tags = next(iter(training_data))
        print(list(prepare_tags(tags).detach().cpu().numpy()))
        print(model(prepare_sentence(sentence)))

    # TRAIN
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(300):
        for i, (sentence, tags) in enumerate(training_data):
            model.zero_grad()
            loss = model.loss(
                sentence=prepare_sentence(sentence),
                tags=prepare_tags(tags),
            )
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch} iter {i}/{len(training_data)} loss {float(loss.detach().cpu().numpy())}")

    with torch.no_grad():
        for sentence, tags in training_data:
            print(list(prepare_tags(tags).detach().cpu().numpy()))
            print(model(prepare_sentence(sentence)))
            print()
