# -*- coding: utf-8 -*-
def precision_recall_f1(y_test, y_predict):
    recalls = []
    precisions = []
    for i in range(len(y_test)):
        recalls.append(sum(y_test[i] * y_predict[i]) / sum(y_test[i]))
        precisions.append(sum(y_test[i] * y_predict[i]) / (sum(y_predict[i])+0.00001))
    recall = sum(recalls) / len(recalls)
    precision = sum(precisions) / len(precisions)
    f1 = 2 * precision * recall / (precision + recall)

    print('precision: %f' % precision)
    print('recall: %f' % recall)
    print('f1: %f' % f1)
    return [precision, recall, f1]

def hamming_loss(y_test, y_predict):
    temp = []
    length = float(len(y_test[0]))
    for i in range(len(y_test)):
        temp.append(sum(y_test[i].astype(int) ^ y_predict[i].astype(int)) / length)
    result = sum(temp) / len(temp)

    print('hamming_loss: %f' % result)
    return result

def micro_macro(y_test, y_predict):
    TPs = []
    FNs = []
    FPs = []
    TNs = []
    for i in range(len(y_test[0])):
        TP = 0.0
        FN = 0.0
        FP = 0.0
        TN = 0.0
        for j in range(len(y_test[:, i])):
            if (y_test[:, i][j] == 1) and (y_predict[:, i][j] == 0):
                FN = FN + 1
            if (y_test[:, i][j] == 1) and (y_predict[:, i][j] == 1):
                TP = TP + 1
            if (y_test[:, i][j] == 0) and (y_predict[:, i][j] == 0):
                TN = TN + 1
            if (y_test[:, i][j] == 0) and (y_predict[:, i][j] == 1):
                FP = FP + 1
        TPs.append(TP)
        FNs.append(FN)
        FPs.append(FP)
        TNs.append(TN)
    micro_precision = (sum(TPs) / len(TPs)) / (sum(TPs) / len(TPs) + sum(FPs) / len(FPs))
    micro_recall = (sum(TPs) / len(TPs)) / (sum(TPs) / len(TPs) + sum(FNs) / len(FNs))
    micro_f1 = 2 * sum(TPs) / len(TPs) / (2 * sum(TPs) / len(TPs) + sum(FPs) / len(FPs) + sum(FNs) / len(FNs))
    print('micro_precision: %f' % micro_precision)
    print('micro_recall: %f' % micro_recall)
    print('micro_f1: %f' % micro_f1)

    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    for i in range(len(TPs)):

        macro_precisions.append(TPs[i] / (TPs[i] + FPs[i] + 0.000000001))
        macro_recalls.append(TPs[i] / (TPs[i] + FNs[i] + 0.000000001))
        macro_f1s.append(2 * TPs[i] / (2 * TPs[i] + FPs[i] + FNs[i] + 0.000000001))


    macro_precision = sum(macro_precisions) / len(macro_precisions)
    macro_recall = sum(macro_recalls) / len(macro_recalls)
    macro_f1 = sum(macro_f1s) / len(macro_f1s)


    print('macro_precision: %f' % macro_precision)
    print('macro_recall: %f' % macro_recall)
    print('macro_f1: %f' % macro_f1)

    return [micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1]

