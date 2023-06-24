


class F1:

    def __init__(self, batch_size, sos_id, eod_id, pad_id, unk_id):
        self.batch_size = batch_size
        self.sos_id = sos_id
        self.eos_id = eod_id
        self.pad_id = pad_id
        self.unk_id = unk_id

    def calculate(self, prediction, target):

        # init metrics
        all_correct = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for i in range(self.batch_size):
            filtered_pred = self.filter_impossible_names(prediction[i, :])
            filtered_trg = self.filter_impossible_names(target[i,:])
            if self.all_match(filtered_pred, filtered_trg):
                all_correct += 1
                true_positive += len(filtered_pred)
            for subtoken in filtered_pred:
                if subtoken in filtered_trg:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtoken in filtered_trg:
                if not subtoken in filtered_pred:
                    false_negative += 1

        # calculate
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        accuracy = all_correct / self.batch_size
        return accuracy, precision, recall, f1

    def filter_impossible_names(self, names_list):
        result = []
        for name in names_list:
            if name not in [self.sos_id, self.eos_id, self.pad_id, self.unk_id]:
                result.append(name)
        return result

    def unique(self, sequence):
        unique = []
        [unique.append(item) for item in sequence if item not in unique]
        return unique

    def all_match(self, pred, trg):
        """
        note that pred & trg must be filtered
        :param pred:
        :param trg:
        :return:
        """
        if pred == trg or self.unique(pred) == self.unique(trg) or ''.join([str(i) for i in pred]) \
                == ''.join([str(i) for i in trg]):
            return True
        else:
            return False
