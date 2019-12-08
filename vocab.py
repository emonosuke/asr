class Vocab:
    def __init__(self, vocab_path):
        with open(vocab_path) as f:
            lines = [line.strip() for line in f.readlines()]

        dic = []
        for line in lines:
            # line: midasi+hinsi ...
            tokens = line.split()
            midasi = tokens[0].split("+")[0]
            dic.append(midasi)
        self.dic = dic

    def id2word(self, idx):
        return self.dic[idx]

    def ids2word(self, idxs):
        return [self.id2word(idx) for idx in idxs]
