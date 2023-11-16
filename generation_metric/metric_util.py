def get_event_pairs(predictions, references):
    event_preds, event_refs = [], []
    minmax_len_list = []
    for pred, ref_list in zip(predictions, references):
        minmax_len = []
        for ref in ref_list:
            min_len = min(len(pred), len(ref))
            max_len = max(len(pred), len(ref))
            minmax_len.append((min_len, max_len))
            for i in range(min_len):
                event_preds.append(pred[i])
                event_refs.append(ref[i])
        minmax_len_list.append(minmax_len)
    return event_preds, event_refs, minmax_len_list


def group_scores(event_score, minmax_len_list, borderline=0):
    start_idx = 0
    score_list, matched_score_list = [], []
    for minmax_len in minmax_len_list:
        cur_max_seq_score, cur_max_matched_seq_score = -1, -1  # max score of current pred compared with different references
        for min_len, max_len in minmax_len:
            end_idx = start_idx + min_len
            mean_of_score = (sum(event_score[start_idx: end_idx]) + borderline * (max_len - min_len)) / max_len
            matched_mean_of_score = sum(event_score[start_idx: end_idx]) / min_len
            start_idx = end_idx
            if mean_of_score > cur_max_seq_score:
                cur_max_seq_score = mean_of_score
            if matched_mean_of_score > cur_max_matched_seq_score:
                cur_max_matched_seq_score = matched_mean_of_score
        score_list.append(cur_max_seq_score)
        matched_score_list.append(cur_max_matched_seq_score)
    return score_list, matched_score_list

