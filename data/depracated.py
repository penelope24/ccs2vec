def align_size(self, arr):
    result = []
    pad_id = self.stoi['[PAD]']
    max_dim = self.max_token_num
    for row in arr:
        if len(row) < max_dim:
            row += [pad_id] * (- len(row))
        elif len(row) > max_dim:
            row = row[:max_dim]
        result.append(row)
    return result