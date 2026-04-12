def get_majority_vote(preds_dict: dict):
    """ Tính phiếu bầu đa số trên một hàng """
    # Lấy danh sách các nhãn dự đoán
    votes = list(preds_dict.values())
    return 4 if votes.count(4) >= 2 else 0

def get_group_vote_batch(mnb_preds, svm_preds, xgb_preds):
    """ Tính phiếu bầu đa số cho toàn bộ danh sách trong một nhóm """

    group_preds = []
    for i in range(len(mnb_preds)):
        votes = {
            "mnb": mnb_preds[i],
            "svm": svm_preds[i],
            "xgb": xgb_preds[i]
        }
        group_preds.append(get_majority_vote(votes))
    return group_preds


def get_track_dual_vote_batch(custom_votes, library_votes):
    """ Xử lý Track-Dual Validation"""
    dual_preds = []
    for cust_vote, lib_vote in zip(custom_votes, library_votes):
        # Ưu tiên Library nếu bất đồng quan điểm
        if cust_vote == lib_vote:
            dual_preds.append(cust_vote)
        else:
            dual_preds.append(lib_vote)
    return dual_preds


def get_track_dual_vote_single(custom_vote, library_vote):
    """
    Phân xử Track-Dual Validation cho 1 văn bản đơn lẻ (dùng trong Inference).
    """
    if library_vote != -1:
        return library_vote if custom_vote != library_vote else custom_vote
    return custom_vote