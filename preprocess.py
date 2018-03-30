import json
import pickle

import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import random

from collections import defaultdict
from functools import partial
from itertools import combinations

def create_co_occurrence_matrix(paper_paper_dict):
    result = defaultdict(partial(defaultdict, int))
    for key, value in paper_paper_dict.items():
        for first, second in combinations(value, 2):
            result[first][second] += 1
            result[second][first] += 1
    result.default_factory = None
    for key in result:
        result[key].default_factory = None
    return result

def create_user_paper_dict(debug=False):
    if debug:
        DBLP_LIST = [ 'dblp-ref/dblp-ref-3.json' ]
    else:
        DBLP_LIST = [ 'dblp-ref/dblp-ref-0.json',
        'dblp-ref/dblp-ref-1.json',
        'dblp-ref/dblp-ref-2.json',
        'dblp-ref/dblp-ref-3.json' ]

    result = defaultdict(partial(defaultdict, int))

    for data in DBLP_LIST:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                
                for author in data["authors"]: # assuming this won't error
                    try:
                        for paper in data ["references"]:
                            result[author][paper] += 1
                    except KeyError:
                        result[author] # this line creates an entry in result
                line = f.readline()
    result.default_factory = None
    for key in result:
        result[key].default_factory = None

    return result

def create_paper_paper_dict(debug=False):
    # It takes about 6 minutes 20 seconds on crunchy5
    if debug:
        DBLP_LIST = [ 'dblp-ref/dblp-ref-3.json' ]
    else:
        DBLP_LIST = [ 'dblp-ref/dblp-ref-0.json',
        'dblp-ref/dblp-ref-1.json',
        'dblp-ref/dblp-ref-2.json',
        'dblp-ref/dblp-ref-3.json' ]

    result = dict()

    for data in DBLP_LIST:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                try:
                    result[data["id"]] = data["references"]
                except KeyError:
                    result[data["id"]] = []
                line = f.readline()

    return result

def save_pickle(save_filename, obj):
    with open(save_filename, 'wb') as handle:
        return pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(save_filename):
    with open(save_filename, 'rb') as handle:
        return pickle.load(handle)

def save_paper_paper_dict():
    # resulting file size about 1G
    # Loading this pickle file actually takes much longer than creating the dictionary!
    # It took me 20 minutes and it still did not finish.
    # I recommend just creating the dict without saving it as pickle.
    save_pickle('dblp-ref/paper_paper_dict.pickle', create_paper_paper_dict())

def save_numbering_and_reverse():
    # TODO: add logic so that only creates these files when they don't exist
    numbering, reverse = assign_number_to_paper_id()
    save_pickle('dblp-ref/numbering.pickle', numbering)
    save_pickle('dblp-ref/reverse.pickle', reverse)

def assign_number_to_paper_id():
    # When using surprise, it seems like we don't need this method.
    # This method assigns nonnegative integer numbers to each paper id,
    # and record it in a dictionary for efficient lookup.
    numbering, reverse = dict(), dict()
    current_id = 0

    for data in DBLP_LIST:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                numbering[data["id"]] = current_id
                reverse[current_id]   = data["id"]
                current_id += 1
                line = f.readline()

    return numbering, reverse

def create_surprise_paper_paper_data(paper_paper_dict, add_random_0_entries=False):
    itemList, userList, ratingList = [], [], []

    all_keys_set = set(paper_paper_dict.keys())
    for key, value in paper_paper_dict.items():
        for paper in value:
            itemList.append(paper)
            userList.append(key)
            ratingList.append(1) # "rating" is always 1 for each citation

        # JP 03/28/18 First attempt on trying to add some (not all) entries with 0 ratings
        if add_random_0_entries and len(value)!=0:
            # create candidate set which does not include references
            zero_rating_set_for_key = all_keys_set - set(value)
            # add randomly selected 0 entry, the same number as 1 entries
            for paper in random.sample(zero_rating_set_for_key, len(value)):
                itemList.append(paper)
                userList.append(key)
                ratingList.append(0) # we add 0 entries (no citation) with certain probability

    ratings_dict = {'itemID': itemList, 'userID': userList, 'rating': ratingList}
    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(0,1)) # JP: rating scale is 0 (not cited) and 1 (cited)

    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    return data

def create_random_subset_paper_paper_data(size=100000, seed=1003, debug=False):
    #Build a random subset of dictionary, where we only retain references to themselves
    mydict = create_paper_paper_dict(debug=debug)
    if size > len(mydict):
        size = len(mydict)
    random.seed(seed)
    random_dict = {k: list(mydict[k]) for k in random.sample(mydict.keys(),size)}
    for each in random_dict:
        for ref in mydict[each]:
            if ref not in random_dict:
                random_dict[each].remove(ref)
    return random_dict


if __name__ == "__main__":
    create_surprise_paper_paper_data(create_random_subset_paper_paper_data(100000,debug=True),True)
