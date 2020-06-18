import collections
import json
import logging
import math
import re
import string

from tqdm import tqdm
from transformers.tokenization_bert import BasicTokenizer

from evaluate import CoQAEvaluator

logger = logging.getLogger(__name__)

CLS_YES = 0
CLS_NO = 1
CLS_UNK = 2
CLS_SPAN = 3

def compute_predictions_logits(all_examples, all_features, all_results, n_best_size,
                               max_answer_length, do_lower_case, output_prediction_file,
                               output_nbest_file, verbose_logging, tokenizer):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index",
            "start_index",
            "end_index",
            "score",
            "cls_idx",
        ])

    all_predictions = []
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(tqdm(all_examples, desc="Writing preditions")):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0

        score_yes, score_no, score_span, score_unk = -float('INF'), -float('INF'), -float('INF'), float('INF')
        min_unk_feature_index, max_yes_feature_index, max_no_feature_index, max_span_feature_index = -1, -1, -1, -1
        # the paragraph slice with min null score

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            # TIP：GET score of y / n / u / s /e
            feature_yes_score, feature_no_score, feature_unk_score = \
                result.yes_logits[0] * 2, result.no_logits[0] * 2, result.unk_logits[0] * 2
            start_indexes, end_indexes = _get_best_indexes(result.start_logits, n_best_size), \
                                         _get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    feature_span_score = result.start_logits[start_index] + result.end_logits[end_index]
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            score=feature_span_score,
                            cls_idx=CLS_SPAN
                        )
                    )

            if feature_unk_score < score_unk:  # find min score_noanswer
                score_unk = feature_unk_score
                min_unk_feature_index = feature_index
            if feature_yes_score > score_yes:  # find max score_yes
                score_yes = feature_yes_score
                max_yes_feature_index = feature_index
            if feature_no_score > score_no:  # find max score_no
                score_no = feature_no_score
                max_no_feature_index = feature_index

        prelim_predictions.append(
            _PrelimPrediction(feature_index=min_unk_feature_index,
                              start_index=0,
                              end_index=0,
                              score=score_unk,
                              cls_idx=CLS_UNK))
        prelim_predictions.append(
            _PrelimPrediction(feature_index=max_yes_feature_index,
                              start_index=0,
                              end_index=0,
                              score=score_yes,
                              cls_idx=CLS_YES))
        prelim_predictions.append(
            _PrelimPrediction(feature_index=max_no_feature_index,
                              start_index=0,
                              end_index=0,
                              score=score_no,
                              cls_idx=CLS_NO))

        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda p: p.score,
                                    reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "score", "cls_idx"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            # including yes/no/noanswer pred
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.cls_idx == CLS_SPAN:
                # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
                nbest.append(_NbestPrediction(text=final_text,score=pred.score,cls_idx=pred.cls_idx))
            else:
                text = ['yes', 'no', 'unknown']
                nbest.append(_NbestPrediction(text=text[pred.cls_idx], score=pred.score, cls_idx=pred.cls_idx))

        if len(nbest) < 1:
            nbest.append(_NbestPrediction(text='unknown', score=-float('inf'), cls_idx=CLS_UNK))

        assert len(nbest) >= 1

        probs = _compute_softmax([p.score for p in nbest])

        nbest_json = []

        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["socre"] = entry.score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        _id, _turn_id = example.qas_id.split()
        all_predictions.append({
            'id': _id,
            'turn_id': int(_turn_id),
            'answer': confirm_preds(nbest_json)
        })

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions

def confirm_preds(nbest_json):
    # Do something for some obvious wrong-predictions
    # TODO: can do more things?
    subs = [
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'true', 'false'
    ]
    ori = nbest_json[0]['text']
    if len(ori) < 2:  # mean span like '.', '!'
        for e in nbest_json[1:]:
            if _normalize_answer(e['text']) in subs:
                return e['text']
        return 'unknown'
    return ori


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" %
                        (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


#=================== For standard evaluation in CoQA =======================

def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = collections.Counter(g_tokens) & collections.Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0

    if len(answers) == 0:
        return 1. if len(pred) == 0 else 0.

    g_tokens = _normalize_answer(pred).split()
    ans_tokens = [_normalize_answer(answer).split() for answer in answers]
    scores = [_score(g_tokens, a) for a in ans_tokens]
    if len(ans_tokens) == 1:
        score = scores[0]
    else:
        score = 0
        for i in range(len(ans_tokens)):
            scores_one_out = scores[:i] + scores[(i + 1):]
            score += max(scores_one_out)
        score /= len(ans_tokens)
    return score


def score(pred, truth, final_json):
    assert len(pred) == len(truth)
    no_ans_total = no_total = yes_total = normal_total = total = 0
    no_ans_f1 = no_f1 = yes_f1 = normal_f1 = f1 = 0
    all_f1s = []
    for p, t, j in zip(pred, truth, final_json):
        total += 1
        this_f1 = _f1_score(p, t)
        f1 += this_f1
        all_f1s.append(this_f1)
        if t[0].lower() == 'no':
            no_total += 1
            no_f1 += this_f1
        elif t[0].lower() == 'yes':
            yes_total += 1
            yes_f1 += this_f1
        elif t[0].lower() == 'unknown':
            no_ans_total += 1
            no_ans_f1 += this_f1
        else:
            normal_total += 1
            normal_f1 += this_f1

    f1 = 100. * f1 / total
    if no_total == 0:
        no_f1 = 0.
    else:
        no_f1 = 100. * no_f1 / no_total
    if yes_total == 0:
        yes_f1 = 0
    else:
        yes_f1 = 100. * yes_f1 / yes_total
    if no_ans_total == 0:
        no_ans_f1 = 0.
    else:
        no_ans_f1 = 100. * no_ans_f1 / no_ans_total
    normal_f1 = 100. * normal_f1 / normal_total
    result = {
        'total': total,
        'f1': f1,
        'no_total': no_total,
        'no_f1': no_f1,
        'yes_total': yes_total,
        'yes_f1': yes_f1,
        'no_ans_total': no_ans_total,
        'no_ans_f1': no_ans_f1,
        'normal_total': normal_total,
        'normal_f1': normal_f1,
    }
    return result, all_f1s


def score_each_instance(pred, truth):
    assert len(pred) == len(truth)
    total = 0
    f1_scores = []
    for p, t in zip(pred, truth):
        total += 1
        f1_scores.append(_f1_score(p, t))
    f1_scores = [100. * x / total for x in f1_scores]
    return f1_scores


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()


def coqa_evaluate(dev_file, output_prediction_file):
    evaluator = CoQAEvaluator(dev_file)

    with open(output_prediction_file) as f:
        pred_data = CoQAEvaluator.preds_to_dict(output_prediction_file)
    return evaluator.model_performance(pred_data)
