from collections import defaultdict

###http://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_items_greater_threshold(predictions, threshold=0):
    '''Return the recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        threshold: recommend the item if est > threshold

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    recommendationlist = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            recommendationlist[uid].append((iid, est))
        else:
            # print('est < threshold')
            pass
    return recommendationlist

# def precision_recall_at_k(predictions, k=10, threshold=3.5):
#     '''Return precision and recall at k metrics for each user.'''

#     # First map the predictions to each user.
#     user_est_true = defaultdict(list)
#     for uid, _, true_r, est, _ in predictions:
#         user_est_true[uid].append((est, true_r))

#     precisions = dict()
#     recalls = dict()
#     for uid, user_ratings in user_est_true.items():

#         # Sort user ratings by estimated value
#         user_ratings.sort(key=lambda x: x[0], reverse=True)

#         # Number of relevant items
#         n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

#         # Number of recommended items in top k
#         n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

#         # Number of relevant and recommended items in top k
#         n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
#                               for (est, true_r) in user_ratings[:k])

#         # Precision@K: Proportion of recommended items that are relevant
#         precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

#         # Recall@K: Proportion of relevant items that are recommended
#         recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

#     return precisions, recalls

def prediction_for_userids(algo, trainset, testdic, userids):
    user_est_true = defaultdict(list)
    for u in userids:
        u_numid = trainset.to_inner_uid(u)
        user_items = set([j for (j, _) in trainset.ur[u_numid]])
        user_est_true[u] = []
        for i in trainset.all_items():
            if i not in user_items :
                if trainset.to_raw_iid(i) in testdic[u]:
                    user_est_true[u].append((algo.estimate(u_numid,i)-algo.trainset.offset,1))
                else:
                    user_est_true[u].append((algo.estimate(u_numid,i)-algo.trainset.offset,0))
    return user_est_true

def precision_recall_at_k(user_est_true, testdict, k=10, threshold=0):
    '''Return precision and recall at k metrics'''
    precisions = dict()
    recalls = dict()
    
    for uid, user_ratings in user_est_true.items():
        
        user_ratings_greater_threshold = [(est, true) for (est,true) in user_ratings if est > threshold]
        
        n_rel = len(testdict[uid])

        n_greater_threshold = len(user_ratings_greater_threshold)
        
        # Number of recommended items in top k
        if n_greater_threshold > k:
            n_rec_k = k
        else:
            n_rec_k = n_greater_threshold
            
        # Number of relevant and recommended items in top k
        if n_greater_threshold > k:
            user_ratings_greater_threshold.sort(key=lambda x: x[0], reverse=True)
            n_rel_and_rec_k = sum((true_r for (est, true_r) in user_ratings_greater_threshold[:k]))
        else:
            try:
                n_rel_and_rec_k = sum((true_r for (est, true_r) in user_ratings_greater_threshold))
            except IndexError:
                n_rel_and_rec_k = 0

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    
    return precisions, recalls
