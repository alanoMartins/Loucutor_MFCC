def dcf_score(y_test, y_pred):
    true_claim = 0
    far = 0
    frr = 0

    for idx in range(0, len(y_test)):
        if int(y_test[idx]) == int(y_pred[idx]):
            true_claim += 1
            continue

        if int(y_test == -1):
            frr += 1

        if int(y_pred == -1):
            far += 1
